import pandas as pd
import tensorflow as tf
from collections import defaultdict
import jieba

WORKING_FOLDER = "."
#####################################################
# Convert Chinese Sentence to Sentence Embedding
#####################################################

str2idx = defaultdict(int)
word2Matrix = []
labels = []
line = 1
maxlen = 500
lstmUnits = 100
batch_size = 8
learning_rate = 0.01
likes = []

tf.flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
tf.flags.DEFINE_integer("max_sentence_length", 100, "Max sentence length in train/test data (Default: 100)")
tf.flags.DEFINE_string("cell_type", 'lstm', "Type of rnn cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)")
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character embedding (Default: 300)")
tf.flags.DEFINE_integer("hidden_size", 128, "Dimensionality of character embedding (Default: 128)")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability (Default: 0.5)")
tf.flags.DEFINE_float("l2_reg_lambda", 3.0, "L2 regularization lambda (Default: 3.0)")
tf.flags.DEFINE_integer("batch_size", batch_size, "Batch Size (Default: 64)")
tf.flags.DEFINE_integer("num_epochs", 100, "Number of training epochs (Default: 100)")
tf.flags.DEFINE_integer("display_every", 10, "Number of iterations to display training info.")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many steps")
tf.flags.DEFINE_integer("num_checkpoints", 5, "Number of checkpoints to store")
tf.flags.DEFINE_float("learning_rate", 1e-3, "Which learning rate to start with. (Default: 1e-3)")

FLAGS = tf.flags.FLAGS
dir(FLAGS)
#FLAGS._parse_flags()

def idx_matrix():                               # open word2vec
    with open('data/sgns.wiki.word') as f:
        reader = csv.reader(f)
        for row in reader:
            str2idx[row[0]] = line
            line += 1
            word2Matrix.append(row[1:])         # construct a matrix for every word in the word2vec
        word2Matrix = np.array(word2Matrix, type = np.float32)
    return str2idx, word2Matrix

def train_embedding():
    str2idx, word2Matrix = idx_matrix()
    with open('data/train.csv', newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            sentence = row[9]
            words = word_lists(sentence, str2idx)
            row_label = row[8]
            row_onehot = [0, 0, 0, 0, 0]
            row_onehot[row_label - 1] = 1
            labels.append(row_onehot)

            likes = row[10]

    return words, labels, str2idx, word2Matrix

def word_lists(sentence, str2idx):
    words = jieba.cut(sentence)
    words = [str2idx[word] for word in words]
    for word in words:
        word.expand([0] * (maxlen - len(word)))
    return words

def train():
    if True:
        # Prepare shuffle data
        words, labels, str2idx, word2Matrix = train_embedding()
        words = np.array(words)
        labels = np.array(labels)
        np.random.seed(10)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x_shuffled = words[shuffle_indices]
        y_shuffled = labels[shuffled_indices]
            
        # Split train/evaluation set
        dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
        x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
        y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
        print("Train/Dev split: {:d}/{:d}\n".format(len(y_train), len(y_dev)))

    with tf.Graph().as_default():
        sess = tf.Session()
        with sess.as_defalut():
            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(FLAGS.max_sentence_length)

            rnn = RNN(
                sequence_length=x_train.shape[1],
                num_classes=5,
                vocab_size=len(text_vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                cell_type=FLAGS.cell_type,
                hidden_size=FLAGS.hidden_size,
                l2_reg_lambda=FLAGS.l2_reg_lambda,
                idx2Matrix = idx2Matrix
            )
            train_op = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(rnn.loss, global_step = global_step)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
            print("Writing to {}\n".format(out_dir))
            
            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", rnn.loss)
            acc_summary = tf.summary.scalar("accuracy", rnn.accuracy)
            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=FLAGS.num_checkpoints)
            sess.run(tf.global_variables_initializer())

            for i in range(len(x_train) // batch_size - 1):
                batch_x = x_train[i * batch_size: (i + 1) * batch_size, :]
                batch_y = y_train[i * batch_size: (i + 1) * batch_size, :]
                batch_x = np.reshape(batch_x, shape = [batch_size, maxlen])
                batch_y = np.reshape(batch_y, shape = [batch_size])
                

                feed_dict = {rnn.input_text: batch_x, rnn.input_y: batch_y, nn.dropout_keep_prob: FLAGS.dropout_keep_prob}
                _, step, summaries, loss, accuracy = sess.run([train_op, global_step, train_summary_op, rnn.loss, rnn.accuracy], feed_dict)

                if step % FLAGS.display_every == 0:
                    time_str = datetime.datetime.now().isoformat()
                    print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if step % FLAGS.evaluate_every == 0:
                    feed_dict_dev = {rnn.input_text: x_dev, rnn.input_y: y_dev, rnn.dropout_keep_prob: 1.0}
                    summaries_dev, loss, accuracy = sess.run([dev_summary_op, rnn.loss, rnn.accuracy], feed_dict_dev)
                    dev_summary_writer.add_summary(summaries_dev, step)

                if step % FLAGS.checkpoint_every == 0:
                    path = saver.save(sess, checkpoint_prefix, global_step=step)


def main(_):
    train()


if __name__ == "__main__":
    tf.app.run()





        





