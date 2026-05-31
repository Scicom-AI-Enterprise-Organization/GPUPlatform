/**
 * Compute waveform peaks from a decoded AudioBuffer for visualization.
 * Returns [min, max] pairs, one per bucket (channel 0).
 */
export function computeWaveformPeaks(
  audioBuffer: AudioBuffer,
  buckets: number,
): [number, number][] {
  const channel = audioBuffer.getChannelData(0);
  const samplesPerBucket = Math.max(1, Math.floor(channel.length / buckets));
  const peaks: [number, number][] = [];
  for (let i = 0; i < buckets; i++) {
    const start = i * samplesPerBucket;
    const end = Math.min(start + samplesPerBucket, channel.length);
    let min = 1;
    let max = -1;
    for (let j = start; j < end; j++) {
      if (channel[j] < min) min = channel[j];
      if (channel[j] > max) max = channel[j];
    }
    peaks.push([min, max]);
  }
  return peaks;
}

// A single shared AudioContext for decoding — browsers cap the number of live
// AudioContexts (~6), and a row browser can mount dozens of players at once.
// decodeAudioData works on a shared (even suspended) context, so one is enough.
let sharedCtx: AudioContext | null = null;
export function decodeContext(): AudioContext | null {
  if (typeof window === "undefined") return null;
  if (!sharedCtx) {
    const Ctor =
      window.AudioContext ||
      (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
    if (!Ctor) return null;
    sharedCtx = new Ctor();
  }
  return sharedCtx;
}
