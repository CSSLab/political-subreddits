// TODO: add total word count to vocabulary, instead of "train_words"
//
// Modifed by Yoav Goldberg, Jan-Feb 2014
// Removed:
//    hierarchical-softmax training
//    cbow
// Added:
//   - support for arbitrary number of vocabularies
//   - different input syntax
//
/////////////////////////////////////////////////////////////////
//
//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include "vocab.h"
#include "io.h"

#define MAX_VOCABS 100
#define MAX_STRING 200
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 1000
#define MAX_CODE_LENGTH 40

typedef float real;                    // Precision of float numbers

char train_file[MAX_STRING], output_file[MAX_STRING];
char vocab_file_base[MAX_STRING];
int binary = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1, use_position = 0;
long long layer1_size = 100;
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
real alpha = 0.025, starting_alpha, sample = 0;
real *syn0, *syn1, *syn1neg, *expTable;
clock_t start;
int numiters = 1;
int num_vocabs = -1;

struct vocabulary *vocabularies[MAX_VOCABS];
real *vecs[MAX_VOCABS];

int negative = 15;
const int table_size = 1e8;
int *unitables[MAX_VOCABS];

long long GetFileSize(char *fname) {
  long long fsize;
  FILE *fin = fopen(fname, "rb");
  if (fin == NULL) {
    printf("ERROR: file not found! %s\n", fname);
    exit(1);
  }
  fseek(fin, 0, SEEK_END);
  fsize = ftell(fin);
  fclose(fin);
  return fsize;
}

// TODO init num_vocabs!

// Used for sampling of negative examples.
// wc[i] == the count of context number i
// wclen is the number of entries in wc (context vocab size)
void InitUnigramTable(int num_vocabs) {
    int a, i, vi;
    long long normalizer = 0;
    struct vocabulary *v;
    real d1, power = 0.75;
    for (vi=0; vi<num_vocabs; ++vi) {
        unitables[vi] = (int *)malloc(table_size * sizeof(int));
        v = vocabularies[vi];
        for (a = 0; a < v->vocab_size; a++) normalizer += pow(v->vocab[a].cn, power);
        i = 0;
        d1 = pow(v->vocab[i].cn, power) / (real)normalizer;
        for (a = 0; a < table_size; a++) {
            unitables[vi][a] = i;
            if (a / (real)table_size > d1) {
                i++;
                d1 += pow(v->vocab[i].cn, power) / (real)normalizer;
            }
            if (i >= v->vocab_size) i = v->vocab_size - 1;
        }
    }
}

void InitNet(int num_vocabs) {
   long long a, b;
   int i;
   for (i=0; i<num_vocabs; ++i) {
       a = posix_memalign((void **)&vecs[i], 128, (long long)vocabularies[i]->vocab_size * layer1_size * sizeof(real));
   if (vecs[i] == NULL) {printf("Memory allocation failed\n"); exit(1);}
   for (b = 0; b < layer1_size; b++) 
      for (a = 0; a < vocabularies[i]->vocab_size; a++)
         vecs[i][a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
   } 
   // TODO add option to Init some embedding tables to 0 instead of Random?
}

// Read word,context pairs from training file, where both word and context are integers.
// We are learning to predict context based on word.
//
// Word and context come from different vocabularies, but we do not really care about that
// at this point.
void *TrainModelThread(void *id) {
   int ctxi = -1, wrdi = -1;
  long long d;
  long long word_count = 0, last_word_count = 0;
  long long l1, l2, c, target, label;
  unsigned long long next_random = (unsigned long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  long long start_offset = file_size / (long long)num_threads * (long long)id;
  long long end_offset = file_size / (long long)num_threads * (long long)(id+1);
  int iter;
  //printf("thread %d %lld %lld \n",id, start_offset, end_offset);
  struct vocabulary *wv;
  struct vocabulary *cv;
  int v1, v2, n1, n2, i, wi, wj;
  char buf[MAX_STRING];
  real *syn0, *syn1neg;
  for (iter=0; iter < numiters; ++iter) {
     fseek(fi, start_offset, SEEK_SET);
     // if not binary:
     while (fgetc(fi) != '\n') { }; //TODO make sure its ok
     printf("thread %d %lld\n", id, ftell(fi));

     long long train_words = 0;
     for (i=0; i < num_vocabs; ++i) { train_words += vocabularies[i]->word_count; } //TODO?
     while (1) { //HERE @@@
         // TODO set alpha scheduling based on number of examples read.
         // The conceptual change is the move from word_count to pair_count
         if (word_count - last_word_count > 10000) {
             word_count_actual += word_count - last_word_count;
             last_word_count = word_count;
             if ((debug_mode > 1)) {
                 now=clock();
                 printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
                         word_count_actual / (real)(numiters*train_words + 1) * 100,
                         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
                 fflush(stdout);
             }
             alpha = starting_alpha * (1 - word_count_actual / (real)(numiters*train_words + 1));
             if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
         }
         if (feof(fi) || ftell(fi) > end_offset) break;
         for (c = 0; c < layer1_size; c++) neu1[c] = 0;
         for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
         int word1_ids[1000]; // TODO constantize
         int word2_ids[1000];

         ReadWord(buf, fi, MAX_STRING); 
         v1 = atoi(buf);
         wv = vocabularies[v1];
         //printf("v1:%d\n", v1);
         n1 = ReadWordsAsIndexes(wv, word1_ids, fi);

         ReadWord(buf, fi, MAX_STRING); 
         v2 = atoi(buf);
         //printf("v2:%d\n", v2);
         cv = vocabularies[v2];
         n2 = ReadWordsAsIndexes(cv, word2_ids, fi);

         syn0 = vecs[v1];
         syn1neg = vecs[v2];
         for (wi=0; wi < n1; ++wi) {
             for (wj=0; wj < n2; ++wj) {
                 wrdi = word1_ids[wi];
                 ctxi = word2_ids[wj];
                 word_count++; //TODO ?
                 if (wrdi < 0 || ctxi < 0) continue;

                 if (sample > 0) {
                     real ran = (sqrt(wv->vocab[wrdi].cn / (sample * wv->word_count)) + 1) * (sample * wv->word_count) / wv->vocab[wrdi].cn;
                     next_random = next_random * (unsigned long long)25214903917 + 11;
                     if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                     ran = (sqrt(cv->vocab[ctxi].cn / (sample * cv->word_count)) + 1) * (sample * cv->word_count) / cv->vocab[ctxi].cn;
                     next_random = next_random * (unsigned long long)25214903917 + 11;
                     if (ran < (next_random & 0xFFFF) / (real)65536) continue;
                 }
                 //fread(&wrdi, 4, 1, fi);
                 //fread(&ctxi, 4, 1, fi);
                 // NEGATIVE SAMPLING
                 l1 = wrdi * layer1_size;
                 for (d = 0; d < negative + 1; d++) {
                     if (d == 0) {
                         target = ctxi;
                         label = 1;
                     } else {
                         next_random = next_random * (unsigned long long)25214903917 + 11;
                         target = unitables[v2][(next_random >> 16) % table_size]; 
                         if (target == 0) target = next_random % (cv->vocab_size - 1) + 1;
                         if (target == ctxi) continue;
                         label = 0;
                     }
                     l2 = target * layer1_size;
                     f = 0;
                     for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
                     if (f > MAX_EXP) g = (label - 1) * alpha;
                     else if (f < -MAX_EXP) g = (label - 0) * alpha;
                     else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
                     for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
                     for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
                 }
                 // Learn weights input -> hidden
                 for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
             }
         }
     }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
}

/* {{{ void *TrainModelThreadOld(void *id) {
  long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
  long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
  long long l1, l2, c, target, label;
  unsigned long long next_random = (long long)id;
  real f, g;
  clock_t now;
  real *neu1 = (real *)calloc(layer1_size, sizeof(real));
  real *neu1e = (real *)calloc(layer1_size, sizeof(real));
  FILE *fi = fopen(train_file, "rb");
  fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
  while (1) {
    if (word_count - last_word_count > 10000) {
      word_count_actual += word_count - last_word_count;
      last_word_count = word_count;
      if ((debug_mode > 1)) {
        now=clock();
        printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
         word_count_actual / (real)(train_words + 1) * 100,
         word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
        fflush(stdout);
      }
      alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
      if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
    }
    if (sentence_length == 0) {
      while (1) {
        word = ReadWordIndex(fi, &wvocabulaty);
        if (feof(fi)) break;
        if (word == -1) continue;
        word_count++;
        if (word == 0) break;
        // The subsampling randomly discards frequent words while keeping the ranking same
        if (sample > 0) {
          real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
          next_random = next_random * (unsigned long long)25214903917 + 11;
          if (ran < (next_random & 0xFFFF) / (real)65536) continue;
        }
        sen[sentence_length] = word;
        sentence_length++;
        if (sentence_length >= MAX_SENTENCE_LENGTH) break;
      }
      sentence_position = 0;
    }
    if (feof(fi)) break;
    if (word_count > train_words / num_threads) break;
    word = sen[sentence_position];
    if (word == -1) continue;
    for (c = 0; c < layer1_size; c++) neu1[c] = 0;
    for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
    //next_random = next_random * (unsigned long long)25214903917 + 11;
    //b = next_random % window;
    // skipgram training
    // b is current window position, in [0,1,...,window-1]
    // word is sen[sentence_position]
    for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
       //printf("b is:%d a:%d\n", (int)b,(int)(a));
       c = sentence_position - window + a;
       if (c < 0) continue;
       if (c >= sentence_length) continue;
       last_word = sen[c];
       if (last_word == -1) continue;
       l1 = last_word * layer1_size;
       for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
       // NEGATIVE SAMPLING
       if (negative > 0) for (d = 0; d < negative + 1; d++) {
          if (d == 0) {
             target = word;
             label = 1;
          } else {
             next_random = next_random * (unsigned long long)25214903917 + 11;
             target = table[(next_random >> 16) % table_size];
             if (target == 0) target = next_random % (vocab_size - 1) + 1;
             if (target == word) continue;
             label = 0;
          }
          if (use_position > 0) {
             l2 = ((a > window?a-1:a) + (window * 2 * target)) * layer1_size;
          } else {
             l2 = target * layer1_size;
          }
          f = 0;
          for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
          if (f > MAX_EXP) g = (label - 1) * alpha;
          else if (f < -MAX_EXP) g = (label - 0) * alpha;
          else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
          for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
          for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
       }
       // Learn weights input -> hidden
       for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
    }
    sentence_position++;
    if (sentence_position >= sentence_length) {
      sentence_length = 0;
      continue;
    }
  }
  fclose(fi);
  free(neu1);
  free(neu1e);
  pthread_exit(NULL);
} }}}*/

void TrainModel() {
  long a, b, c, d;
  int i;
  FILE *fo;
  struct vocabulary *wv;
  file_size = GetFileSize(train_file);
  pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
  printf("Starting training using file %s\n", train_file);
  starting_alpha = alpha;
  for (i=0; i<MAX_VOCABS; ++i) vocabularies[i]=0;
  for (i=0; i<num_vocabs; ++i) {
      char vocab_file[200]; // The filename buffer.
      snprintf(vocab_file, sizeof(char) * 200, "%s%i", vocab_file_base, i);
      vocabularies[i]=ReadVocab(vocab_file);
  }
  //wv = ReadVocab(wvocab_file); // TODO read vocabularies from file
  //cv = ReadVocab(cvocab_file);
  InitNet(num_vocabs);
  InitUnigramTable(num_vocabs);
  start = clock();
  for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
  for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
  // TODO num_vocabs output files
  for (i=0; i<num_vocabs; ++i) {
      char output_file_name[200];
      snprintf(output_file_name, sizeof(char) * 200, "%s%i", output_file, i);
      fo = fopen(output_file_name, "wb");
      wv = vocabularies[i];
      syn0 = vecs[i];
      if (classes == 0) { // TODO get rid of classes (and cmdline option)
          fprintf(fo, "%d %d\n", wv->vocab_size, layer1_size);
          for (a = 0; a < wv->vocab_size; a++) {
              fprintf(fo, "%s ", wv->vocab[a].word); //TODO
              if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
              else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
              fprintf(fo, "\n");
          }
      }
      fclose(fo);
  }
}

int ArgPos(char *str, int argc, char **argv) {
  int a;
  for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
    if (a == argc - 1) {
      printf("Argument missing for %s\n", str);
      exit(1);
    }
    return a;
  }
  return -1;
}

int main(int argc, char **argv) {
  int i;
  if (argc == 1) {
    printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
    printf("Options:\n");
    printf("Parameters for training:\n");
    printf("\t-train <file>\n");
    printf("\t\tUse text data from <file> to train the model\n");
    printf("\t-output <file>\n");
    printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
    printf("\t-size <int>\n");
    printf("\t\tSet size of word vectors; default is 100\n");
    printf("\t-negative <int>\n");
    printf("\t\tNumber of negative examples; default is 15, common values are 5 - 10 (0 = not used)\n");
    printf("\t-threads <int>\n");
    printf("\t\tUse <int> threads (default 1)\n");
    //printf("\t-min-count <int>\n");
    //printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
    printf("\t-sample <float>\n");
    printf("\t\tSet threshold for occurrence of words and contexts. Those that appear with higher frequency");
    printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value in the original word2vec was 1e-5\n");
    printf("\t-alpha <float>\n");
    printf("\t\tSet the starting learning rate; default is 0.025\n");
    printf("\t-iters <int>\n");
    printf("\t\tPerform i iterations over the data; default is 1\n");
    printf("\t-classes <int>\n");
    printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
    printf("\t-binary <int>\n");
    printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
    printf("\t-numv n\n");
    printf("\t\tNumber of vocabularies.\n");
    printf("\t-vocab filename\n");
    printf("\t\tvocabulary base file name (will read filename.0, filename.1, etc\n");
    printf("\nExamples:\n");
    printf("./word2vecf -train data.txt -vocab wv -output vec.txt -size 200 -negative 5 -threads 10 \n\n");
    return 0;
  }
  output_file[0] = 0;
  vocab_file_base[0] = 0;
  if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-vocab", argc, argv)) > 0) strcpy(vocab_file_base, argv[i + 1]);
  if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
  if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
  if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
  if ((i = ArgPos((char *)"-iters", argc, argv)) > 0) numiters = atoi(argv[i+1]);
  if ((i = ArgPos((char *)"-numv", argc, argv)) > 0) num_vocabs = atoi(argv[i+1]);

  if (output_file[0] == 0) { printf("must supply -output.\n\n"); return 0; }
  if (num_vocabs < 0) { printf("must supply -numv.\n\n"); return 0; }
  if (vocab_file_base[0] == 0) { printf("must supply -vocab.\n\n"); return 0; }
  expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
  for (i = 0; i < EXP_TABLE_SIZE; i++) {
    expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
    expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
  }
  TrainModel();
  return 0;
}
