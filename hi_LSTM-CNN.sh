# @Author: feidong1991
# @Date:   2017-01-09 17:01:51
# @Last Modified by:   feidong1991
# @Last Modified time: 2017-06-18 16:48:02


# Script to train neural model on automatic text scoring
checkdir=./checkpoint

if [ "$#" -ne 4 ]; then
  echo "Usage: sh $0 embed_dim prompt_id fold_id embed_type"
  exit 1
fi


embed_dim=$1
prompt_id=$2
fold_id=$3
embed_type=$4

echo $embed_type

datadir=../baselines/nea/data/fold_${fold_id}
trainfile=$datadir/train.tsv
devfile=$datadir/dev.tsv
testfile=$datadir/test.tsv

if [ ! -d $checkdir/preds ]; then
	mkdir -p $checkdir/preds
fi

if [ $embed_type = "word2vec" ]
then
	embed_dir=../data/embedding
	embeddingfile=$embed_dir/data_dim${embed_dim}.vec
elif [ $embed_type = "glove" ]
then
	embed_dir=../data/embedding/glove.6B
	embeddingfile=$embed_dir/glove.6B.${embed_dim}d.txt.gz
fi

nb_epochs=50
# echo $embed_dim

echo "Using embedding ${embeddingfile}"

 # THEANO_FLAGS='floatX=float32,device=cpu'
 python hi_LSTM-CNN.py --fine_tune --embedding $embed_type --embedding_dict $embeddingfile --embedding_dim ${embed_dim} \
 	--num_epochs $nb_epochs --batch_size 10 --nbfilters 100 --filter1_len 5 --filter2_len 3 --rnn_type LSTM --lstm_units 100\
 	--optimizer rmsprop --learning_rate 0.001 --dropout 0.5  \
	--oov embedding  --checkpoint_path $checkdir \
	--train $trainfile --dev $devfile --test $testfile --prompt_id $prompt_id --train_flag --mode att #--init_bias #--l2_value 0.001
