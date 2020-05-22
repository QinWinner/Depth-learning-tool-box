# -*- coding: utf-8 -*-
import sys

import webrtcvad
import collections
import sys
import pyaudio

from array import array
from struct import pack
import wave
import time

def vad1():
	FORMAT = pyaudio.paInt16
	CHANNELS = 1
	RATE = 16000
	CHUNK_DURATION_MS = 20       # supports 10, 20 and 30 (ms)   #检测时间
	PADDING_DURATION_MS = 1500   # 1 sec jugement  #空白时间
	CHUNK_SIZE = int(RATE * CHUNK_DURATION_MS / 1000)  # chunk to read
	CHUNK_BYTES = CHUNK_SIZE * 2  # 16bit = 2 bytes, PCM
	NUM_PADDING_CHUNKS = int(PADDING_DURATION_MS / CHUNK_DURATION_MS)
# NUM_WINDOW_CHUNKS = int(240 / CHUNK_DURATION_MS)
	NUM_WINDOW_CHUNKS = int(400 / CHUNK_DURATION_MS)  # 400 ms/ 30ms  ge 13
	NUM_WINDOW_CHUNKS_END = NUM_WINDOW_CHUNKS * 2 #26

	START_OFFSET = int(NUM_WINDOW_CHUNKS * CHUNK_DURATION_MS * 0.5 * RATE)

	vad = webrtcvad.Vad(3)

	pa = pyaudio.PyAudio()
	stream = pa.open(format=FORMAT,
                 channels=CHANNELS,
                 rate=RATE,
                 input=True,
                 start=False,
                 # input_device_index=2,
                 frames_per_buffer=CHUNK_SIZE)


	got_a_sentence1 = False
	leave1 = False


	def handle_int(sig, chunk):
		global leave1, got_a_sentence1
		leave1 = True
		got_a_sentence1 = True


	def record_to_file(path, data, sample_width):
		"Records from the microphone and outputs the resulting data to 'path'"
    # sample_width, data = record()
		data = pack('<' + ('h' * len(data)), *data)
		wf = wave.open(path, 'wb')
		wf.setnchannels(1)
		wf.setsampwidth(sample_width)
		wf.setframerate(RATE)
		wf.writeframes(data)
		wf.close()


	def normalize(snd_data):
		"Average the volume out"
		MAXIMUM = 32767  # 16384
		times = float(MAXIMUM) / max(abs(i) for i in snd_data)
		r = array('h')
		for i in snd_data:
			r.append(int(i * times))
		return r

	i = 0
	while not leave1:
		ring_buffer = collections.deque(maxlen=NUM_PADDING_CHUNKS)  #双端队列
		triggered = False
		point = False
		voiced_frames = []
		ring_buffer_flags = [0] * NUM_WINDOW_CHUNKS
		ring_buffer_index = 0

		ring_buffer_flags_end = [0] * NUM_WINDOW_CHUNKS_END
		ring_buffer_index_end = 0
		buffer_in = ''
		# WangS
		#raw_data = array('h')
		raw_data = array("h")
		index = 0
		start_point = 0
		StartTime = time.time()
		print("* 麦克已开启: ")
		stream.start_stream()

		while not got_a_sentence1 and not leave1:
			chunk = stream.read(CHUNK_SIZE)

        # add WangS
			raw_data.extend(array("h",chunk))
			index += CHUNK_SIZE
			
			TimeUse = time.time() - StartTime
			active = vad.is_speech(chunk, RATE)

			sys.stdout.write('1' if active else '_')
			ring_buffer_flags[ring_buffer_index] = 1 if active else 0
			ring_buffer_index += 1
			ring_buffer_index %= NUM_WINDOW_CHUNKS

			ring_buffer_flags_end[ring_buffer_index_end] = 1 if active else 0
			ring_buffer_index_end += 1
			ring_buffer_index_end %= NUM_WINDOW_CHUNKS_END

        # start point detection
			if not triggered:
				ring_buffer.append(chunk)
				num_voiced = sum(ring_buffer_flags)
				if num_voiced > 0.85 * NUM_WINDOW_CHUNKS:
					sys.stdout.write(' Open ')

					triggered = True
					point = True
					start_point = index - CHUNK_SIZE * 20  # start point
                # voiced_frames.extend(ring_buffer)
					ring_buffer.clear()
			# end point detection
			else:
            # voiced_frames.append(chunk)
				ring_buffer.append(chunk)
				num_unvoiced = NUM_WINDOW_CHUNKS_END - sum(ring_buffer_flags_end)
				#print (num_unvoiced)
				if num_unvoiced > 0.85 * NUM_WINDOW_CHUNKS_END or TimeUse > 15:
					sys.stdout.write(' Close ')
					triggered = False
					got_a_sentence1 = True

			sys.stdout.flush()

		sys.stdout.write('\n')
		# data = b''.join(voiced_frames)
		stream.stop_stream()
		print("* 麦克已关闭")
		got_a_sentence1= False
		# write to file
		raw_data.reverse()
		for index in range(start_point):
			raw_data.pop()
		raw_data.reverse()
		raw_data = normalize(raw_data)
		record_to_file("assets/01.wav", raw_data, 2)
		leave1 = True

	stream.stop_stream()
	stream.close()
	pa.terminate()

	return 1
	
	
