
Producing embeddings with word2vec_mc:
====

There are three stages:

      1. Create input data, which is in the form of:

        <vocab_id> item1 item2 ... itemk ||| <vocab_id> item1 item2 ... itemj

        where <vocab_id> is a number (0-based) of the vocabulary/embedding-matrix of this group of items.
        each item from the first group is considered a "word", while each item from the second group
        is considered a "context".
        for each (itemi,itemj) pair, there will be n negative samples, in which itemj' is sampled from the corresponding
        vocabulary.

      2. Create the actual vocabularies:

            python mc_extract_vocabs.py --min_count 20 train_file vocab_prefix

         This will go over the words in train_file and write the corresponding vocabularies into 
         vocab_prefix0 vocab_prefix1 ....
      
      3. Train the embeddings:
         
            ./myword2vec/word2vec_mc -train train_file -vocab vocab_prefix -numv 4 -output dim200vecs. -size 200 -negative 15 -threads 10

         This will train 200-dim embeddings based on 
         not in `cv` are ignored).
         
         The -dumpcv flag can be used in order to dump the trained context-vectors as well.

            ./myword2vec/word2vecf -train dep.contexts -wvocab wv -cvocab cv -output dim200vecs -size 200 -negative 15 -threads 10 -dumpcv dim200context-vecs


      3.5 convert the embeddins to numpy-readable format:
         
            ./scripts/vecs2nps.py dim200vecs vecs

          This will create `vecs.npy` and `vecs.vocab`, which can be read by
          the infer.py script.
