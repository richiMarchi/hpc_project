/* Riccardo Marchi - MAT: 0000753342 */

#include "hpc.h"
#include <stdio.h>
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

/* Puts the original bmap in a larger bmap in order to
   add ghost cells on all sides */
void toextended(int R, bmap_t *cur, bmap_t *next) {

  next->n = cur->n + 2 * R;
  next->bmap = (cell_t*)malloc((next->n) * (next->n) * sizeof(cell_t));

  int i, j;

  /* Sets the original bmap at the center of the larger one */
  for(i = 0; i < cur->n; i++) {
    for(j = 0; j < cur->n; j++) {
      *IDX(next->bmap, next->n, i + R, j + R) = *IDX(cur->bmap, cur->n, i, j);
    }
  }
}

/* Puts the modified bmap in a smaller bmap without ghost cells
   in order to export it to a pbm image */
void toreduced(int R, bmap_t *next, bmap_t *cur) {

  int i, j;

  /* Takes the center of the larger bmap excluding ghost cells */
  for(i = 0; i < cur->n; i++) {
    for(j = 0; j < cur->n; j++) {
      *IDX(cur->bmap, cur->n, i, j) = *IDX(next->bmap, next->n, i + R, j + R);
    }
  }
}

/* Sets the ghost cells on each side */
void copy_ghost(int radius, bmap_t *ltl) {

  int i,j;
  int n = ltl->n;
  int noRad = n - 2 * radius;

  #pragma omp parallel for private(i, j) collapse(2)
  for (i = 0; i < n; i++) {
    for (j = 0; j < n; j++) {

      /* upper side ghost cells */
      if (i < radius && j >= radius && j <= n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i + noRad, j);
      } else

      /* lower side ghost cells */
      if (i > n - radius - 1 && j >= radius && j <= n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i - noRad, j);
      } else

      /* left side ghost cells */
      if (j < radius && i >= radius && i <= n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i, j + noRad);
      } else

      /* right side ghost cells */
      if (j > n - radius - 1 && i >= radius && i <= n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i, j - noRad);
      } else

      /* upper-left side ghost cells */
      if (i < radius && j < radius) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i + noRad, j + noRad);
      } else

      /* lower-left side ghost cells */
      if (i > n - radius - 1 && j < radius) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i - noRad, j + noRad);
      } else

      /* upper-right side ghost cells */
      if (i < radius && j > n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i + noRad, j - noRad);
      } else

      /* lower-right side ghost cells */
      if (i > n - radius - 1 && j > n - radius - 1) {
        *IDX(ltl->bmap, n, i, j) = *IDX(ltl->bmap, n, i - noRad, j - noRad);
      }
     }
   }
}

/* Execute a step of the computation, from cur to stepped */
void step(int radius, bmap_t *cur, bmap_t *stepped, int B1, int B2, int D1, int D2) {

  int sum, i, j, s, f;
  int n = cur->n;

  #pragma omp parallel for private(i, j, s, f, sum) collapse(2)
  /* calculate each element of the source matrix */
  for (i = radius; i <= n - radius - 1; i++) {
    for (j = radius; j <= n - radius - 1; j++) {

      sum = 0;
      cell_t selected = *IDX(cur->bmap, n, i, j);

      /* calculate alive neighbors (selected element included!!!) */
      for (s = i - radius; s <= i + radius; s++) {
        for (f = j - radius; f <= j + radius; f++) {
          sum += *IDX(cur->bmap, n, s, f);
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
    double exe_time;
    const char *infile, *outfile;
    FILE *in, *out;
    bmap_t cur, next, temp;
    bmap_t *tonext, *totemp, *tp;

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

    /* Creates a larger matrix to add ghost on each side */
    toextended(R, &cur, &next);

    temp.n = next.n;
    temp.bmap = (cell_t*)malloc((temp.n) * (temp.n) * sizeof(cell_t));

    /* Assign pointers to manage matrixes */
    tonext = &next;
    totemp = &temp;

    exe_time = hpc_gettime();

    /* Each step ghost cells are copied and the computation is made */
    for(ns = 0; ns < nsteps; ns++) {
      copy_ghost(R, tonext);
      step(R, tonext, totemp, B1, B2, D1, D2);

      /* If it is not the last step the pointers are updated */
      if (ns != nsteps - 1) {
        tp = totemp;
        totemp = tonext;
        tonext = tp;
      }
    }

    exe_time = hpc_gettime() - exe_time;

    /* Put the result in the original matrix */
    toreduced(R, totemp, &cur);

    out = fopen(outfile, "w");
    if ( out == NULL ) {
        fprintf(stderr, "FATAL: can not open \"%s\" for writing", outfile);
        exit(-1);
    }
    write_ltl(&cur, out);
    fclose(out);

    printf("Execution Time: %f\n", exe_time);

    free(cur.bmap);
    free(next.bmap);
    free(temp.bmap);

    return 0;
}
