import numpy as np
import tensorflow as tf
import scipy.io.wavfile as wav
import os
import sys
sys.path.append("DeepSpeech")
import random
from scipy.signal import butter, lfilter


###########################################################################
# This section of code is credited to:
## Copyright (C) 2017, Nicholas Carlini <nicholas@carlini.com>.

# Okay, so this is ugly. We don't want DeepSpeech to crash.
# So we're just going to monkeypatch TF and make some things a no-op.
# Sue me.
tf.load_op_library = lambda x: x
generation_tmp = os.path.exists
os.path.exists = lambda x: True


class Wrapper:
    def __init__(self, d):
        self.d = d

    def __getattr__(self, x):
        return self.d[x]


class HereBeDragons:
    d = {}
    FLAGS = Wrapper(d)

    def __getattr__(self, x):
        return self.do_define

    def do_define(self, k, v, *x):
        self.d[k] = v


tf.app.flags = HereBeDragons()
import DeepSpeech
os.path.exists = generation_tmp

# More monkey-patching, to stop the training coordinator setup
DeepSpeech.TrainingCoordinator.__init__ = lambda x: None
DeepSpeech.TrainingCoordinator.start = lambda x: None

from DeepSpeech.util.text import ctc_label_dense_to_sparse
from tf_logits import compute_mfcc, get_logits

# These are the tokens that we're allowed to use.
# The - token is special and corresponds to the epsilon
# value in CTC decoding, and can not occur in the phrase.
toks = " abcdefghijklmnopqrstuvwxyz'-"

# Importer
# ========
tf.app.flags.DEFINE_string  ('train_files',      '',          'comma separated list of files specifying the dataset used for training. multiple files will get merged')
tf.app.flags.DEFINE_string  ('dev_files',        '',          'comma separated list of files specifying the dataset used for validation. multiple files will get merged')
tf.app.flags.DEFINE_string  ('test_files',       '',          'comma separated list of files specifying the dataset used for testing. multiple files will get merged')
tf.app.flags.DEFINE_boolean ('fulltrace',        False,       'if full trace debug info should be generated during training')

# Cluster configuration
# =====================
tf.app.flags.DEFINE_string  ('ps_hosts',         '',          'parameter servers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('worker_hosts',     '',          'workers - comma separated list of hostname:port pairs')
tf.app.flags.DEFINE_string  ('job_name',         'localhost', 'job name - one of localhost (default), worker, ps')
tf.app.flags.DEFINE_integer ('task_index',       0,           'index of task within the job - worker with index 0 will be the chief')
tf.app.flags.DEFINE_integer ('replicas',         -1,          'total number of replicas - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('replicas_to_agg',  -1,          'number of replicas to aggregate - if negative, its absolute value is multiplied by the number of workers')
tf.app.flags.DEFINE_integer ('coord_retries',    100,         'number of tries of workers connecting to training coordinator before failing')
tf.app.flags.DEFINE_string  ('coord_host',       'localhost', 'coordination server host')
tf.app.flags.DEFINE_integer ('coord_port',       2500,        'coordination server port')
tf.app.flags.DEFINE_integer ('iters_per_worker', 1,           'number of train or inference iterations per worker before results are sent back to coordinator')

# Global Constants
# ================
tf.app.flags.DEFINE_boolean ('train',            True,        'whether to train the network')
tf.app.flags.DEFINE_boolean ('test',             True,        'whether to test the network')
tf.app.flags.DEFINE_integer ('epoch',            75,          'target epoch to train - if negative, the absolute number of additional epochs will be trained')

tf.app.flags.DEFINE_boolean ('use_warpctc',      False,       'whether to use GPU bound Warp-CTC')

tf.app.flags.DEFINE_float   ('dropout_rate',     0.05,        'dropout rate for feedforward layers')
tf.app.flags.DEFINE_float   ('dropout_rate2',    -1.0,        'dropout rate for layer 2 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate3',    -1.0,        'dropout rate for layer 3 - defaults to dropout_rate')
tf.app.flags.DEFINE_float   ('dropout_rate4',    0.0,         'dropout rate for layer 4 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate5',    0.0,         'dropout rate for layer 5 - defaults to 0.0')
tf.app.flags.DEFINE_float   ('dropout_rate6',    -1.0,        'dropout rate for layer 6 - defaults to dropout_rate')

tf.app.flags.DEFINE_float   ('relu_clip',        20.0,        'ReLU clipping value for non-recurrant layers')

# Adam optimizer (http://arxiv.org/abs/1412.6980) parameters
tf.app.flags.DEFINE_float   ('beta1',            0.9,         'beta 1 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('beta2',            0.999,       'beta 2 parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('epsilon',          1e-8,        'epsilon parameter of Adam optimizer')
tf.app.flags.DEFINE_float   ('learning_rate',    0.001,       'learning rate of Adam optimizer')

# Batch sizes
tf.app.flags.DEFINE_integer ('train_batch_size', 1,           'number of elements in a training batch')
tf.app.flags.DEFINE_integer ('dev_batch_size',   1,           'number of elements in a validation batch')
tf.app.flags.DEFINE_integer ('test_batch_size',  1,           'number of elements in a test batch')

tf.app.flags.DEFINE_integer ('export_batch_size', 1,          'number of elements per batch on the exported graph')

# Performance (UNSUPPORTED)
tf.app.flags.DEFINE_integer ('inter_op_parallelism_threads', 0, 'number of inter-op parallelism threads - see tf.ConfigProto for more details')
tf.app.flags.DEFINE_integer ('intra_op_parallelism_threads', 0, 'number of intra-op parallelism threads - see tf.ConfigProto for more details')

# Sample limits
tf.app.flags.DEFINE_integer ('limit_train',      0,           'maximum number of elements to use from train set - 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_dev',        0,           'maximum number of elements to use from validation set- 0 means no limit')
tf.app.flags.DEFINE_integer ('limit_test',       0,           'maximum number of elements to use from test set- 0 means no limit')

# Step widths
tf.app.flags.DEFINE_integer ('display_step',     0,           'number of epochs we cycle through before displaying detailed progress - 0 means no progress display')
tf.app.flags.DEFINE_integer ('validation_step',  0,           'number of epochs we cycle through before validating the model - a detailed progress report is dependent on "--display_step" - 0 means no validation steps')

# Checkpointing
tf.app.flags.DEFINE_string  ('checkpoint_dir',   '',          'directory in which checkpoints are stored - defaults to directory "deepspeech/checkpoints" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('checkpoint_secs',  600,         'checkpoint saving interval in seconds')
tf.app.flags.DEFINE_integer ('max_to_keep',      5,           'number of checkpoint files to keep - default value is 5')

# Exporting
tf.app.flags.DEFINE_string  ('export_dir',       '',          'directory in which exported models are stored - if omitted, the model won\'t get exported')
tf.app.flags.DEFINE_integer ('export_version',   1,           'version number of the exported model')
tf.app.flags.DEFINE_boolean ('remove_export',    False,       'whether to remove old exported models')
tf.app.flags.DEFINE_boolean ('use_seq_length',   True,        'have sequence_length in the exported graph (will make tfcompile unhappy)')
tf.app.flags.DEFINE_integer ('n_steps',          16,          'how many timesteps to process at once by the export graph, higher values mean more latency')

# Reporting
tf.app.flags.DEFINE_integer ('log_level',        1,           'log level for console logs - 0: INFO, 1: WARN, 2: ERROR, 3: FATAL')
tf.app.flags.DEFINE_boolean ('log_traffic',      False,       'log cluster transaction and traffic information during debug logging')
tf.app.flags.DEFINE_boolean ('show_progressbar', True,        'Show progress for training, validation and testing processes. Log level should be > 0.')

tf.app.flags.DEFINE_string  ('wer_log_pattern',  '',          'pattern for machine readable global logging of WER progress; has to contain %%s, %%s and %%f for the set name, the date and the float respectively; example: "GLOBAL LOG: logwer(\'12ade231\', %%s, %%s, %%f)" would result in some entry like "GLOBAL LOG: logwer(\'12ade231\', \'train\', \'2017-05-18T03:09:48-0700\', 0.05)"; if omitted (default), there will be no logging')

tf.app.flags.DEFINE_boolean ('log_placement',    False,       'whether to log device placement of the operators to the console')
tf.app.flags.DEFINE_integer ('report_count',     10,          'number of phrases with lowest WER (best matching) to print out during a WER report')

tf.app.flags.DEFINE_string  ('summary_dir',      '',          'target directory for TensorBoard summaries - defaults to directory "deepspeech/summaries" within user\'s data home specified by the XDG Base Directory Specification')
tf.app.flags.DEFINE_integer ('summary_secs',     0,           'interval in seconds for saving TensorBoard summaries - if 0, no summaries will be written')

# Geometry
tf.app.flags.DEFINE_integer ('n_hidden',         2048,        'layer width to use when initialising layers')

# Initialization
tf.app.flags.DEFINE_integer ('random_seed',      4567,        'default random seed that is used to initialize variables')

# Early Stopping
tf.app.flags.DEFINE_boolean ('early_stop',       True,        'enable early stopping mechanism over validation dataset. Make sure that dev FLAG is enabled for this to work')

# This parameter is irrespective of the time taken by single epoch to complete and checkpoint saving intervals.
# It is possible that early stopping is triggered far after the best checkpoint is already replaced by checkpoint saving interval mechanism.
# One has to align the parameters (earlystop_nsteps, checkpoint_secs) accordingly as per the time taken by an epoch on different datasets.
tf.app.flags.DEFINE_integer ('earlystop_nsteps',  4,          'number of steps to consider for early stopping. Loss is not stored in the checkpoint so when checkpoint is revived it starts the loss calculation from start at that point')
tf.app.flags.DEFINE_float   ('estop_mean_thresh', 0.5,        'mean threshold for loss to determine the condition if early stopping is required')
tf.app.flags.DEFINE_float   ('estop_std_thresh',  0.5,        'standard deviation threshold for loss to determine the condition if early stopping is required')

# Decoder
tf.app.flags.DEFINE_string  ('decoder_library_path', 'native_client/libctc_decoder_with_kenlm.so', 'path to the libctc_decoder_with_kenlm.so library containing the decoder implementation.')
tf.app.flags.DEFINE_string  ('alphabet_config_path', 'data/alphabet.txt', 'path to the configuration file specifying the alphabet used by the network. See the comment in data/alphabet.txt for a description of the format.')
tf.app.flags.DEFINE_string  ('lm_binary_path',       'data/lm/lm.binary', 'path to the language model binary file created with KenLM')
tf.app.flags.DEFINE_string  ('lm_trie_path',         'data/lm/trie', 'path to the language model trie file created with native_client/generate_trie')
tf.app.flags.DEFINE_integer ('beam_width',        1024,       'beam width used in the CTC decoder when building candidate transcriptions')
tf.app.flags.DEFINE_float   ('lm_weight',         1.75,       'the alpha hyperparameter of the CTC decoder. Language Model weight.')
tf.app.flags.DEFINE_float   ('valid_word_count_weight', 1.00, 'valid word insertion weight. This is used to lessen the word insertion penalty when the inserted word is part of the vocabulary.')

# Inference mode
tf.app.flags.DEFINE_string  ('one_shot_infer',       '',       'one-shot inference mode: specify a wav file and the script will load the checkpoint and perform inference on it. Disables training, testing and exporting.')

# Initialize from frozen model
tf.app.flags.DEFINE_string  ('initialize_from_frozen_model', '', 'path to frozen model to initialize from. This behaves like a checkpoint, loading the weights from the frozen model and starting training with those weights. The optimizer parameters aren\'t restored, so remember to adjust the learning rate.')

FLAGS = tf.app.flags.FLAGS

###########################################################################


def db(audio):
    if len(audio.shape) > 1:
        maxx = np.max(np.abs(audio), axis=1)
        return 20 * np.log10(maxx) if np.any(maxx != 0) else np.array([0])
    maxx = np.max(np.abs(audio))
    return 20 * np.log10(maxx) if maxx != 0 else np.array([0])


def load_wav(input_wav_file):
    # Load the inputs that we're given
    fs, audio = wav.read(input_wav_file)
    assert fs == 16000
    print('source dB', db(audio))
    return audio


def save_wav(audio, output_wav_file):
    wav.write(output_wav_file, 16000, np.array(np.clip(np.round(audio), -2**15, 2**15-1), dtype=np.int16))
    print('output dB', db(audio))


def levenshteinDistance(s1, s2): 
    if len(s1) > len(s2):
        s1, s2 = s2, s1
        
    distances = range(len(s1) + 1) 
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]


def highpass_filter(data, cutoff=7000, fs=16000, order=10):
    b, a = butter(order, cutoff / (0.5 * fs), btype='high', analog=False)
    return lfilter(b, a, data)


def get_new_pop(elite_pop, elite_pop_scores, pop_size):
    scores_logits = np.exp(elite_pop_scores - elite_pop_scores.max()) 
    elite_pop_probs = scores_logits / scores_logits.sum()
    cand1 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    cand2 = elite_pop[np.random.choice(len(elite_pop), p=elite_pop_probs, size=pop_size)]
    mask = np.random.rand(pop_size, elite_pop.shape[1]) < 0.5 
    next_pop = mask * cand1 + (1 - mask) * cand2
    return next_pop


def mutate_pop(pop, mutation_p, noise_stdev, elite_pop):
    noise = np.random.randn(*pop.shape) * noise_stdev
    noise = highpass_filter(noise)
    mask = np.random.rand(pop.shape[0], elite_pop.shape[1]) < mutation_p 
    new_pop = pop + noise * mask
    return new_pop


class Genetic():
    def __init__(self, input_wave_file, output_wave_file, target_phrase):
        self.pop_size = 100
        self.elite_size = 10
        self.mutation_p = 0.005
        self.noise_stdev = 40
        self.noise_threshold = 1
        self.mu = 0.9
        self.alpha = 0.001
        self.max_iters = 3000
        self.num_points_estimate = 100
        self.delta_for_gradient = 100
        self.delta_for_perturbation = 1e3
        self.input_audio = load_wav(input_wave_file).astype(np.float32)
        self.pop = np.expand_dims(self.input_audio, axis=0)
        self.pop = np.tile(self.pop, (self.pop_size, 1))
        self.output_wave_file = output_wave_file
        self.target_phrase = target_phrase
        self.funcs = self.setup_graph(self.pop, np.array([toks.index(x) for x in target_phrase]))

    def setup_graph(self, input_audio_batch, target_phrase): 
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)
        
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            
            inputs = tf.placeholder(tf.float32, shape=pass_in.shape, name='a')
            len_batch = tf.placeholder(tf.float32, name='b')
            arg2_logits = tf.placeholder(tf.int32, shape=logits_arg2.shape, name='c')
            arg1_dense = tf.placeholder(tf.float32, shape=dense_arg1.shape, name='d')
            arg2_dense = tf.placeholder(tf.int32, shape=dense_arg2.shape, name='e')
            len_seq = tf.placeholder(tf.int32, shape=seq_len.shape, name='f')
            
            logits = get_logits(inputs, arg2_logits)
            target = ctc_label_dense_to_sparse(arg1_dense, arg2_dense, len_batch)
            ctcloss = tf.nn.ctc_loss(labels=tf.cast(target, tf.int32), inputs=logits, sequence_length=len_seq)
            decoded, _ = tf.nn.ctc_greedy_decoder(logits, arg2_logits, merge_repeated=True)
            
            sess = tf.Session()
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, "models/session_dump")
            
        func1 = lambda a, b, c, d, e, f: sess.run(ctcloss, 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        func2 = lambda a, b, c, d, e, f: sess.run([ctcloss, decoded], 
            feed_dict={inputs: a, len_batch: b, arg2_logits: c, arg1_dense: d, arg2_dense: e, len_seq: f})
        return (func1, func2)

    def getctcloss(self, input_audio_batch, target_phrase, decode=False):
        batch_size = input_audio_batch.shape[0]
        weird = (input_audio_batch.shape[1] - 1) // 320 
        logits_arg2 = np.tile(weird, batch_size)
        dense_arg1 = np.array(np.tile(target_phrase, (batch_size, 1)), dtype=np.int32)
        dense_arg2 = np.array(np.tile(target_phrase.shape[0], batch_size), dtype=np.int32)
        
        pass_in = np.clip(input_audio_batch, -2**15, 2**15-1)
        seq_len = np.tile(weird, batch_size).astype(np.int32)

        if decode:
            return self.funcs[1](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        else:
            return self.funcs[0](pass_in, batch_size, logits_arg2, dense_arg1, dense_arg2, seq_len)
        
    def get_fitness_score(self, input_audio_batch, target_phrase, input_audio, classify=False):
        target_enc = np.array([toks.index(x) for x in target_phrase])
        final_text = ''
        if classify:
            ctcloss, decoded = self.getctcloss(input_audio_batch, target_enc, decode=True)
            all_text = "".join([toks[x] for x in decoded[0].values]) 
            index = len(all_text) // input_audio_batch.shape[0] 
            final_text = all_text[:index]
        else:
            ctcloss = self.getctcloss(input_audio_batch, target_enc)
        score = -ctcloss
        if classify:
            return score, final_text
        return score, -ctcloss
                    
    def run(self, log=None):
        max_fitness_score = float('-inf') 
        dist = float('inf')
        best_text = ''
        itr = 1
        prev_loss = None
        if log is not None:
            log.write('target phrase: ' + self.target_phrase + '\n')
            log.write('itr, corr, lev dist \n')
        
        while itr <= self.max_iters and best_text != self.target_phrase:
            pop_scores, ctc = self.get_fitness_score(self.pop, self.target_phrase, self.input_audio)
            elite_ind = np.argsort(pop_scores)[-self.elite_size:]
            elite_pop, elite_pop_scores, elite_ctc = self.pop[elite_ind], pop_scores[elite_ind], ctc[elite_ind]
            
            if prev_loss is not None and prev_loss != elite_ctc[-1]: 
                self.mutation_p = self.mu * self.mutation_p + self.alpha / np.abs(prev_loss - elite_ctc[-1]) 

            if itr % 10 == 0:
                print('**************************** ITERATION {} ****************************'.format(itr))
                print('Current loss: {}'.format(-elite_ctc[-1]))
                save_wav(elite_pop[-1], self.output_wave_file)
                best_pop = np.tile(np.expand_dims(elite_pop[-1], axis=0), (100, 1))
                _, best_text = self.get_fitness_score(best_pop, self.target_phrase, self.input_audio, classify=True)
                
                dist = levenshteinDistance(best_text, self.target_phrase)
                corr = "{0:.4f}".format(np.corrcoef([self.input_audio, elite_pop[-1]])[0][1])
                print('Audio similarity to input: {}'.format(corr))
                print('Edit distance to target: {}'.format(dist))
                print('Currently decoded as: {}'.format(best_text))
                if log is not None:
                    log.write(str(itr) + ", " + corr + ", " + str(dist) + "\n")
                    
            if dist > 2:
                next_pop = get_new_pop(elite_pop, elite_pop_scores, self.pop_size)
                self.pop = mutate_pop(next_pop, self.mutation_p, self.noise_stdev, elite_pop)
                prev_loss = elite_ctc[-1]
                
            else:
                perturbed = np.tile(np.expand_dims(elite_pop[-1], axis=0), (self.num_points_estimate, 1))
                indices = np.random.choice(self.pop.shape[1], size=self.num_points_estimate, replace=False)

                perturbed[np.arange(self.num_points_estimate), indices] += self.delta_for_gradient
                perturbed_scores = self.get_fitness_score(perturbed, self.target_phrase, self.input_audio)[0]

                grad = (perturbed_scores - elite_ctc[-1]) / self.delta_for_gradient
                grad /= np.abs(grad).max()
                modified = elite_pop[-1].copy()
                modified[indices] += grad * self.delta_for_perturbation

                self.pop = np.tile(np.expand_dims(modified, axis=0), (self.pop_size, 1))
                self.delta_for_perturbation *= 0.995
                
            itr += 1

        return itr > self.max_iters


inp_wav_file = sys.argv[1]
target = sys.argv[2].lower()
out_wav_file = inp_wav_file[:-4] + '_adv.wav'
log_file = inp_wav_file[:-4] + '_log.txt'

print('target phrase:', target)
print('source file:', inp_wav_file)

g = Genetic(inp_wav_file, out_wav_file, target)
with open(log_file, 'w') as log:
    success = g.run(log=log)

if success:
    print('Succes! Wav file stored as', out_wav_file)
else:
    print('Not totally a success! Consider running for more iterations. Intermediate output stored as', out_wav_file)
