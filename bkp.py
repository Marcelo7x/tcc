figureOfTime = 1
beats = 4
going = 60

beatTime = 60.0/going
compassTime = 4 * figureOfTime * beatTime

windowSize = args.blocksize / args.samplerate
windowPerBeat = beatTime / windowSize

twoCompasse = 2 * beats * windowPerBeat

for i in range(beats):
    print(beats - i)
    time.sleep(figureOfTime * beatTime)