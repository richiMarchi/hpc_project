/* Riccardo Marchi - MAT: 0000753342 */

#include "hpc.h"
#include <stdio.h>
#include <mpi.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <ctype.h> /* for isdigit */

typedef unsigned char cell_t;

/* This struct is defined here as an example; it is possible to modify
   this definition, or remove it altogether if you prefer to pass
   around the pointer to the bitmap directly. */
typedef struct {
    int n;
    cell_t *bmap;
} bmap_t;

/* Returns a pointer to the cell of coordinates (i,j) in the bitmap
   bmap */
cell_t *IDX(cell_t *bmap, int n, int i, int j)
{
    return bmap + i*n + j;
}

/**
 * Write the content of the bmap_t structure pointed to by ltl to the
 * file f in PBM format. The caller is responsible for passing a
 * pointer f to a file opened for writing
 */
void write_ltl( bmap_t* ltl, FILE *f )
{
    int i, j;
    const int n = ltl->n;

    fprintf(f, "P1\n");
    fprintf(f, "# produced by ltl\n");
    fprintf(f, "%d %d\n", n, n);
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            fprintf(f, "%d ", *IDX(ltl->bmap, n, i, j));
        }
        fprintf(f, "\n");
    }
}

/**
 * Read a PBM file from file f. The caller is responsible for passing
 * a pointer f to a file opened for reading. This function is not very
 * robust; it may fail on perfectly legal PBM images, but should work
 * for the images produced by gen-input.c. Also, it should work with
 * PBM images produced by Gimp (you must save them in "ASCII format"
 * when prompted).
 */
void read_ltl( bmap_t *ltl, FILE* f )
{
    char buf[2048];
    char *s;
    int n, i, j;
    int width, height;

    /* Get the file type (must be "P1") */
    s = fgets(buf, sizeof(buf), f);
    if (0 != strcmp(s, "P1\n")) {
        fprintf(stderr, "FATAL: Unsupported file type \"%s\"\n", buf);
        exit(-1);
    }
    /* Get any comment and ignore it; does not work if there are
       leading spaces in the comment line */
    do {
        s = fgets(buf, sizeof(buf), f);
    } while (s[0] == '#');
    /* Get width, height; since we are assuming square images, we
       reject the input if width != height. */
    sscanf(s, "%d %d", &width, &height);
    if ( width != height ) {
        fprintf(stderr, "FATAL: image width (%d) and height (%d) must be equal\n", width, height);
        exit(-1);
    }
    ltl->n = n = width;
    ltl->bmap = (cell_t*)malloc( n * n * sizeof(cell_t));
    /* scan bitmap; each pixel is represented by a single numeric
       character ('0' or '1'); spaces and other separators are ignored
       (Gimp produces PBM files with no spaces between digits) */
    for (i=0; i<n; i++) {
        for (j=0; j<n; j++) {
            int val;
            do {
                val = fgetc(f);
                if ( EOF == val ) {
                    fprintf(stderr, "FATAL: error reading input\n");
                    exit(-1);
                }
            } while ( !isdigit(val) );
            *IDX(ltl->bmap, n, i, j) = (val - '0');
        }
    }
}

/* Execute a step of the computation, from cur to stepped */
void step(int radius, bmap_t *cur, bmap_t *stepped, bmap_t *up, bmap_t *low, int chunk, int B1, int B2, int D1, int D2) {

  int sum, i, j, s, f;
  int n = cur->n;

  /* calculate each element of the source matrix */
  for (i = 0; i < chunk; i++) {
    for (j = 0; j < n; j++) {

      sum = 0;
      cell_t selected = *IDX(cur->bmap, n, i, j);

      /* calculate alive neighbors (selected element included!!!) */
      for (s = i - radius; s <= i + radius; s++) {
        for (f = j - radius; f <= j + radius; f++) {
          //Upper-left side
          if (s < 0 && f < 0) {
            sum += *IDX(up->bmap, n, radius + s, n + f);
          } else
          //Upper-right side
          if (s < 0 && f >= n) {
            sum += *IDX(up->bmap, n, radius + s, f - n);
          } else
          //Lower-left side
          if (s >= chunk && f < 0) {
            sum += *IDX(low->bmap, n, s - chunk, n + f);
          } else
          //Lower-right side
          if (s >= chunk && f >= n) {
            sum += *IDX(low->bmap, n, s - chunk, f - n);
          } else
          //Upper side
          if (s < 0 && f >= 0 && f < n) {
            sum += *IDX(up->bmap, n, radius + s, f);
          } else
          //Lower side
          if (s >= chunk && f >= 0 && f < n) {
            sum += *IDX(low->bmap, n, s - chunk, f);
          } else
          //Left side
          if (s >= 0 && s < chunk && f < 0) {
            sum += *IDX(cur->bmap, n, s, n + f);
          } else
          //Right side
          if (s >= 0 && s < chunk && f >= n) {
            sum += *IDX(cur->bmap, n, s, f - n);
          } else
          //Any other inside cell
          {
            sum += *IDX(cur->bmap, n, s, f);
          }
        }
      }

      /* changes or confirms cell's status according to input rules */
      if (selected == 0 && sum >= B1 && sum <= B2) {
        *IDX(stepped->bmap, n, i, j) = 1;
      } else if (selected == 1 && !(sum >= D1 && sum <= D2)) {
        *IDX(stepped->bmap, n, i, j) = 0;
      } else {
        *IDX(stepped->bmap, n, i, j) = selected;
      }
    }
  }
}

int main( int argc, char* argv[] )
{
    int R, B1, B2, D1, D2, nsteps, ns;
    int rank, size;
    int previous, following;
    int chunksize, width;
    double exe_time;
    const char *infile, *outfile;
    FILE *in, *out;
    bmap_t cur, procMap, stepMap, upperSide, lowerSide;
    bmap_t *tonext, *totemp, *tp, *upSide, *lowSide;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Status status;

    if ( argc != 9 ) {
        fprintf(stderr, "Usage: %s R B1 B2 D1 D2 nsteps infile outfile\n", argv[0]);
        return -1;
    }
    R = atoi(argv[1]);
    B1 = atoi(argv[2]);
    B2 = atoi(argv[3]);
    D1 = atoi(argv[4]);
    D2 = atoi(argv[5]);
    nsteps = atoi(argv[6]);
    infile = argv[7];
    outfile = argv[8];

    assert(  R <= 8  );
    assert(  0 <= B1 );
    assert( B1 <= B2 );
    assert(  1 <= D1 );
    assert( D1 <= D2 );

    /* Only the master loads the input image and checks if the length of the side
     is multiple of the number of processors */
    if (rank == 0) {
      in = fopen(infile, "r");
      if (in == NULL) {
          fprintf(stderr, "FATAL: can not open \"%s\" for reading\n", infile);
          exit(-1);
      }
      read_ltl(&cur, in);
      fclose(in);

      fprintf(stderr, "Size of input image: %d x %d\n", cur.n, cur.n);
      fprintf(stderr, "Model parameters: R=%d B1=%d B2=%d D1=%d D2=%d nsteps=%d\n",
              R, B1, B2, D1, D2, nsteps);

      if (cur.n % size != 0) {
        fprintf(stderr, "The side of the matrix must be multiple of %d\n", size);
        MPI_Abort(MPI_COMM_WORLD, 0);
      } else {
        chunksize = cur.n / size;
        width = cur.n;
      }
    }

    /* Set each processor's previous and following task */
    if (rank == 0) {
      previous = size - 1;
    } else {
      previous = rank - 1;
    }
    if (rank == size - 1) {
      following = 0;
    } else {
      following = rank + 1;
    }

    /* Each task gets the size of the local block from the master */
    MPI_Bcast(&chunksize, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /* Each task allocates memory blocks for the computation and for ghost cells from other tasks */
    procMap.bmap = (cell_t*)malloc(chunksize * width * sizeof(cell_t));
    procMap.n = width;
    tonext = &procMap;
    stepMap.bmap = (cell_t*)malloc(chunksize * width * sizeof(cell_t));
    stepMap.n = width;
    totemp = &stepMap;
    upperSide.bmap = (cell_t*)malloc(width * R * sizeof(cell_t));
    upperSide.n = width;
    upSide = &upperSide;
    lowerSide.bmap = (cell_t*)malloc(width * R * sizeof(cell_t));
    lowerSide.n = width;
    lowSide = &lowerSide;

    /* The master divides the main image in equal parts and sends each part to the
    corresponding task */
    MPI_Scatter(cur.bmap, chunksize * width, MPI_CHAR,
                tonext->bmap, chunksize * width, MPI_CHAR, 0, MPI_COMM_WORLD);

    /* Tasks exchange their upper part of the matrix to the previous one and get
    the lower side of ghost cell from it */
    MPI_Sendrecv(tonext->bmap, R * width, MPI_CHAR, previous, 0,
                 lowSide->bmap, width * chunksize, MPI_CHAR, following, 0, MPI_COMM_WORLD, &status);
    /* Tasks exchange their lower part of the matrix to the following one and get
    the upper side of ghost cell from it */
    MPI_Sendrecv(IDX(tonext->bmap, procMap.n, chunksize - R, 0), R * width, MPI_CHAR, following, 0,
                 upSide->bmap, width * chunksize, MPI_CHAR, previous, 0, MPI_COMM_WORLD, &status);

    exe_time = hpc_gettime();

    for(ns = 0; ns < nsteps; ns++) {

      /* Computation of the step */
      step(R, tonext, totemp, upSide, lowSide, chunksize, B1, B2, D1, D2);

      /* Exchange of blocks pointers */
      tp = totemp;
      totemp = tonext;
      tonext = tp;

      /* If it's not at the end, the tasks exchange ghost cells */
      if (ns != nsteps - 1) {
        MPI_Sendrecv(tonext->bmap, R * width, MPI_CHAR, previous, 0,
                     lowSide->bmap, width * chunksize, MPI_CHAR, following, 0, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(IDX(tonext->bmap, procMap.n, chunksize - R, 0), R * width, MPI_CHAR, following, 0,
                     upSide->bmap, width * chunksize, MPI_CHAR, previous, 0, MPI_COMM_WORLD, &status);
      }
    }

    /* At the end of the computation the master gets all the parts and recreates a unique image */
    MPI_Gather(tonext->bmap, chunksize * width, MPI_CHAR, cur.bmap, chunksize * width, MPI_CHAR, 0, MPI_COMM_WORLD);

    exe_time = hpc_gettime() - exe_time;

    /* Only the master writes out the final matrix to file */
    if (rank == 0) {
      out = fopen(outfile, "w");
      if ( out == NULL ) {
          fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
          exit(-1);
      }
      write_ltl(&cur, out);
      fclose(out);

      printf("Execution Time: %f\n", exe_time);

      free(cur.bmap);
    }

    free(procMap.bmap);
    free(stepMap.bmap);
    free(upperSide.bmap);
    free(lowerSide.bmap);

    MPI_Finalize();

    return 0;
}
