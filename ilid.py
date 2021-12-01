import tensorflow as tf

batch_size = 64
raw_data_ds = tf.data.TextLineDataset(["pickup_lines.txt"])

text = ""
for elem in raw_data_ds:
    text = text + (elem.numpy().decode('utf-8'))

splitted = tf.strings.bytes_split(text)
print("Corpus length:", int(len(text)/1000), "K chars")
chars = sorted(list(set(text)))
print("Total disctinct chars:", len(chars))

maxlen = 32
step = 3
input_chars = []
next_char = []
for i in range(0, len(text) - maxlen, step):
    input_chars.append(text[i: i + maxlen])
    next_char.append(text[i + maxlen])

X_train_ds_raw=tf.data.Dataset.from_tensor_slices(input_chars)
y_train_ds_raw=tf.data.Dataset.from_tensor_slices(next_char)

def custom_standardization(input_data):
    lowercase     = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    stripped_num  = tf.strings.regex_replace(stripped_html, "[\d-]", " ")
    stripped_punc  =tf.strings.regex_replace(stripped_num, 
                             "[%s]" % re.escape(string.punctuation), "")    
    return stripped_punc

def char_split(input_data):
  return tf.strings.unicode_split(input_data, 'UTF-8')
  
# Model constants.
max_features = 91              # Number of distinct chars / words  
embedding_dim = 16             # Embedding layer output dimension
sequence_length = maxlen       # Input sequence size

vectorize_layer = tf.keras.layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    split=char_split, # word_split or char_split
    output_mode="int",
    output_sequence_length=sequence_length,
)

vectorize_layer.adapt(X_train_ds_raw.batch(batch_size))

def vectorize_text(text):
  text = tf.expand_dims(text, -1)
  return tf.squeeze(vectorize_layer(text))

X_train_ds = X_train_ds_raw.map(vectorize_text)
y_train_ds = y_train_ds_raw.map(vectorize_text)

X_train_ds.element_spec, y_train_ds.element_spec

y_train_ds=y_train_ds.map(lambda x: x[0])

for elem in y_train_ds.take(1):
  print("shape: ", elem.shape, "\n next_char: ",elem.numpy())

train_ds =  tf.data.Dataset.zip((X_train_ds,y_train_ds))

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.shuffle(buffer_size=512).batch(batch_size, drop_remainder=True).cache().prefetch(buffer_size=AUTOTUNE)