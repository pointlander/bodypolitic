#!/usr/bin/env python3

import fire
import json
import os
import numpy as np
import tensorflow as tf
import re
import random

import model, sample, encoder

class Node:
    name = ""
    topic = ""
    bias = ""
    state = [""] * 3
    index = 0

    def __init__(self, name, topic, bias):
        self.name = name
        self.topic = topic
        self.bias = bias

    def context(self):
        ctx = self.bias
        i = self.index
        while True:
            ctx += " " + self.state[i]
            i = (i + 1) % 3
            if i == self.index:
                break
        ctx += " " + self.topic
        return ctx.strip()

    def say(self, message):
        print(self.name + ": " + message)

    def listen(self, message):
        i = self.index
        self.state[i] = message
        i = (i + 1) % 3
        self.index = i

def interact_model(
    model_name='117M',
    seed=None,
    nsamples=1,
    batch_size=1,
    length=64,
    temperature=1,
    top_k=40,
):
    """
    Interactively run the model
    :model_name=117M : String, which model to use
    :seed=None : Integer seed for random number generators, fix seed to reproduce
     results
    :nsamples=1 : Number of samples to return total
    :batch_size=1 : Number of batches (only affects speed/memory).  Must divide nsamples.
    :length=None : Number of tokens in generated text, if None (default), is
     determined by model hyperparameters
    :temperature=1 : Float value controlling randomness in boltzmann
     distribution. Lower temperature results in less random completions. As the
     temperature approaches zero, the model will become deterministic and
     repetitive. Higher temperature results in more random completions.
    :top_k=0 : Integer value controlling diversity. 1 means only 1 word is
     considered for each step (token), resulting in deterministic completions,
     while 40 means 40 words are considered at each step. 0 (default) is a
     special setting meaning no restrictions. 40 generally is a good value.
    """
    if batch_size is None:
        batch_size = 1
    assert nsamples % batch_size == 0

    enc = encoder.get_encoder(model_name)
    hparams = model.default_hparams()
    with open(os.path.join('models', model_name, 'hparams.json')) as f:
        hparams.override_from_dict(json.load(f))

    if length is None:
        length = hparams.n_ctx // 4
    elif length > hparams.n_ctx:
        raise ValueError("Can't get samples longer than window size: %s" % hparams.n_ctx)

    with tf.Session(graph=tf.Graph()) as sess:
        context = tf.placeholder(tf.int32, [batch_size, None])
        np.random.seed(seed)
        tf.set_random_seed(seed)
        output = sample.sample_sequence(
            hparams=hparams, length=length,
            context=context,
            batch_size=batch_size,
            temperature=temperature, top_k=top_k
        )

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(os.path.join('models', model_name))
        saver.restore(sess, ckpt)
        def sampleModel(raw_text):
            context_tokens = enc.encode(raw_text)
            generated = 0
            for _ in range(nsamples // batch_size):
                out = sess.run(output, feed_dict={
                    context: [context_tokens for _ in range(batch_size)]
                })[:, len(context_tokens):]
                for i in range(batch_size):
                    generated += 1
                    text = enc.decode(out[i])
                    position = re.search(r"[.!?]", text)
                    if position is None:
                        return text
                    else:
                        position = position.start() + 1
                        return text[:position].strip()

        topic = "Build the wall:"
        nodes = []
        nodes.append(Node("Trump", topic, "Donald Trump is great!"))
        nodes.append(Node("Clinton", topic, "Hillary Clinton is great!"))
        nodes.append(Node("Sanders", topic, "Bernie Sanders is great!"))
        last = -1
        selected = 0
        while True:
            while True:
                selected = random.randint(0,2)
                if selected != last:
                    last = selected
                    break
            ctx = nodes[selected].context()
            message = sampleModel(ctx)
            nodes[selected].say(message)
            print()
            for i in range(len(nodes)):
                if i != selected:
                    nodes[i].listen(message)

if __name__ == '__main__':
    fire.Fire(interact_model)
