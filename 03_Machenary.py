#
#	03_Machenary
#

import tensorflow as tf
import numpy as np
import keras
import random as r
import os

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense

import wikipediaapi

import itertools as i

vocab = 10000

# Criando um tokenizer
tokenizer = keras.preprocessing.text.Tokenizer(num_words=10000,  filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~1234567890')

prompts = []
responses = []

diretorio = "/storage/emulated/0/direct/"
conteudo_diretorio = os.listdir(diretorio)

itens=0

max_art = 0

for item in conteudo_diretorio:
	itens += 1
	if itens == None:
		break
	pc = (open(diretorio + item+ '/abstract.txt').read())
	rc = (open(diretorio + item+ '/main.txt').read())
	prompts.append(pc)
	responses.append(rc)
	if max_art < len(rc):
		max_art = len(rc)

tokenizer.fit_on_texts(prompts)
tokenizer.fit_on_texts(responses)

vocab = len(tokenizer.word_index) + 1

def str_combo(combo):
	r = ""
	for k in combo:
		r += k + " "
	return r
	
wiki_wiki = wikipediaapi.Wikipedia('wikipedia project', 'en')

def add_page(name):
	try:
		page_py = wiki_wiki.page(name)
		print("Page - Title: %s" % page_py.title)
		print("Page - Summary: %s" % page_py.summary)
		return page_py.summary
	except:
		return ""

textos = []
rotulos = []

def noise_page(z):
	print(z)
	z = z.split(" ")
	for n in range(len(z)):
		if r.randint(1,10) > 7:
			vocab_size = len(tokenizer.word_index)
			random_index = r.randint(1, vocab_size)
			random_word = tokenizer.index_word[random_index]
			z[n] = random_word
	h = ""
	for w in z:
		h += " " + w
	return h

def noise_page_step(l, n):
	for k in range(n):
		text = noise_page(l)
	return l

for j in range(2):
	for k in i.product("abcdefghijklmnopqrstuvwxyz", repeat=j):
		g = add_page(str_combo(k))
		if len(g) > 0 and g is not '' and g is not None:
			textos.append(g)
			rotulos.append(1)
			z = 6
			for n in range(6):
				z -= 1
				textos.append(noise_page_step(g, n))
				rotulos.append(z * 0.1)




for prompt, response in zip(prompts, responses):
	textos.append(prompt)
	rotulos.append(r.randint(1,5) * .1)
	textos.append(responses)
	rotulos.append(r.randint(1,5) * .1)

# fedding the classifer

tokenizer_ = Tokenizer(num_words=1000)

tokenizer_.fit_on_texts(prompts)
tokenizer_.fit_on_texts(responses)
tokenizer_.fit_on_texts(textos)

tokenizer.fit_on_texts(textos)

for k in range(100):
	sentence = ""
	for n in range(vocab):
		vocab_size = len(tokenizer.word_index)
		random_index = r.randint(1, vocab_size)
		random_word = tokenizer.index_word[random_index]
		sentence += random_word + " "
	textos.append(sentence)
	rotulos.append(.1)

print(textos, rotulos)

sequences = tokenizer_.texts_to_sequences(textos)
X = pad_sequences(sequences, vocab)

classifier = Sequential()
classifier.add(Embedding(input_dim=1000, output_dim=64, input_length=X.shape[1]))
classifier.add(Flatten())
classifier.add(Dense(1, activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

print(classifier.summary())

classifier.fit(X, np.array(rotulos), epochs=200, batch_size=32)

def classific(text):
	global vocab
	global tokenizer_
	inp = tokenizer_.texts_to_sequences([text])
	inp = pad_sequences(inp, vocab)
	return classifier.predict(inp)[0]
	
text0 = "This is a time machine structure"
text1 = "A random random random rat pig bee horse text."
text2 = "1 5 7 2 4 6"

print(classific(text0), text0)
print(classific(text1), text1)
print(classific(text2), text2)

print('vacaboulary', vocab, 'max article', max_art)

def normalize(text):
	text = tokenizer.texts_to_sequences([text])
	text = tf.keras.preprocessing.sequence.pad_sequences(text, vocab)
	return text

def normalize_(text, vocab):
	text = tokenizer.texts_to_sequences([text])
	text = tf.keras.preprocessing.sequence.pad_sequences(text, vocab)
	return text
	

def reverse(tokens):
	return tokenizer.sequences_to_texts([tokens])

def generate(n, model, generated):
	for n in range(n):
		prediction = model.predict(normalize(generated))
		print(prediction, np.argmax([prediction]))
		revers = reverse(np.array([np.argmax(prediction)]))
		generated += ' ' + revers[0]
		print(generated, "classific -> ", classific(generated))
	return generated

model = tf.keras.Sequential([
tf.keras.layers.Embedding(10000, 128),
tf.keras.layers.LSTM(512),
tf.keras.layers.Dense(vocab, activation='softmax')], name='Generative_model')
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary(), 'summary generative')

denoiser = tf.keras.Sequential([
tf.keras.layers.Input(shape=(vocab,)),
tf.keras.layers.Dense(vocab, activation='relu'),
], name='Denoisification_model')

print(denoiser.summary(), 'summary denoiser')

print(denoiser.input_shape)
print(denoiser.output_shape)

# Compile the model
denoiser.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

x = np.array([])
y = np.array([])


#
#	method for transfer articles and prompts into geberative model
#
counter = 0
article_indice = 0
for prompt, response in zip(prompts, responses):
	article_indice += 1
	result = prompt
	counter_ = 0
	for w in response.split(" "):
		counter_ += 1
		nm = tokenizer.texts_to_sequences([w])
		if len(nm[0]) == 1:
			print(w, nm, article_indice, 'ok')
			nm = tf.keras.preprocessing.sequence.pad_sequences(nm, 1)[0]
			x = np.append(x, normalize(result)[0])
			z = np.zeros(vocab)
			z[nm -1] = 1
			y = np.append(y, z)
			result += ' ' + w
			counter += 1
			if counter_ == 1:
				break

#	noise method
def noise(inp):
	for n in range(len(inp)):
		if inp[n] > 0 and r.randint(1, 10) <= 3:
			inp[n] += int(r.randint(-10, 10))
	return inp

def mutate(model):
	model = tf.keras.Sequential([
	tf.keras.layers.Embedding(10000, r.randint(1, 512)),
	tf.keras.layers.LSTM(r.randint(1, 512)),
	tf.keras.layers.Dense(vocab, activation='softmax')], name='Generative_model')
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

x = x.reshape(counter, vocab)
y = y.reshape(counter, vocab)

model.fit(x, y, epochs=1)

# implementa transfer learning or mutation schema

mutation = True
tune = False

mach_prompts = [
		"infinity energy machine",
		"time machine",
		"who is you"
]
mach_responses = [
		"syntheticly structure to generate inifinity amount of energy",
		"synthetic structure to travel syntheticly trougth the time",
		"i am generative structure data model"
]

tokenizer.fit_on_texts(mach_prompts)
tokenizer.fit_on_texts(mach_responses)

print(len(mach_prompts), len(mach_responses))

if len(mach_prompts) is not len(mach_responses):
	print('quiting model may have the same shape for mach prompts and mach responses')
	exit()

max_mach_prompt = 0
max_mach_resp = 0

for k in mach_prompts:
	if max_mach_prompt < len(k):
		max_mach_prompt = len(k)

for k in mach_responses:
	if max_mach_resp < len(k):
		max_mach_resp = len(k)

max_mach = max_mach_prompt + max_mach_resp

envolved = False
es = 0 							# evolve steps

config_classific =  0.8 # classifier we are ensured that the model is predicting with high quality so dont worries
config_similarity = 0.7 # similarity betwen prompt and response to the generated data
config_mean_model = 0.0 # mean of acertivity of all prompts

def similarity(gen, text):
	counter = 0
	for word in gen.split(" "):
		if word in text:
			counter += 1
	return counter / len(gen.split(" "))

if mutation:
	while envolved == False:
		es += 1
		acc = 0
		indice = 0
		mutate(model)
		for prompt, response in zip(mach_prompts, mach_responses):
			indice += 1
			gen = generate(max_mach, model, prompt)
			print(gen, 'debug')
			c = classific(str(gen))
			print(c,"debug")
			s = similarity(gen, response)
			if c >= config_classific and s >= config_similarity:
				acc += 1
			print('generation step', indice, c, "%  classification generated ->", s, "% similarity")
		mean = acc / len(mach_prompts)
		if mean > config_mean_model:
			model.save("pefect_model.h5")
			envolved = True
		print(es, " try left to envolve", mean,'%')


dx = np.array([])
dy = np.array([])

counter = 0

# work to left here maybe in the future near we will work on this above

#
#	mehod for fedding the denoising model
#
print('fedding denoising network')
for prompt in prompts:
	steps = 1000
	article_x = normalize(generate(100, model, prompt))[0]
	for n in range(steps):
		print('denoising step', n)
		counter += 1
		nc = noise(article_x)
		dx = np.append(dx, article_x)
		dy = np.append(dy, nc)
		article_x = nc


dx = dx.reshape(counter, vocab)
dy = dy.reshape(counter, vocab)

denoiser.fit(dx, dy, epochs=5)

generated = generate(100, model, "infinity energy machine")

denoised = normalize(generated)[0]

for n in denoised:
	if n > 0:
		print(n, 'denoised normalized')

#		above method for denoising and data reconstruction
for n in range(100):
	denoised = noise(denoised);

print(denoised)
print(denoised.shape)

denoised = denoised.reshape(1, vocab)

for n in range(100):
	denoised = denoiser.predict(denoised).astype(int)
	r = reverse(denoised[0])
	print(r)

