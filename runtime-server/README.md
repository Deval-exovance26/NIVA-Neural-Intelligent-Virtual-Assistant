# runtime-server/

The **live runtime** of NIVA (Arc B). Nova Sonic speech-to-speech session service +
WebRTC transport, co-located with the offline build on the single EC2 g6e.4xlarge.
This service streams **audio + text + derived FLAME params** to the browser; it never
renders avatar video.

## Role (per docs/architecture.md)

```
Mic ─► WebRTC ─► Nova Sonic (InvokeModelWithBidirectionalStream)
                      │
                      ├─► response audio ──────────────► Speaker (client)
                      │        │
                      │        ▼ Audio→FLAME (ARTalk-style, 25fps: expr/jaw/pose/eye)
                      │        │
                      │        ▼ FLAME param frames ─(data channel)─► browser web-renderer
                      ▼
             (A/V sync owned client-side)
```

Responsibilities:
- Open `InvokeModelWithBidirectionalStream` to Amazon Bedrock Nova Sonic; handle the
  structured event sequence (completionStart → audio output). Treat as **audio-out**;
  Nova Sonic emits audio + text only (no phoneme visemes).
- Run the **ARTalk-style audio→FLAME** driver (25fps FLAME motioncode) to derive lip/jaw/
  pose/eye motion from the audio stream.
- Serve WebRTC: mic up; Nova Sonic audio + derived FLAME params down (data channel for
  FLAME frames time-aligned to audio).
- Emit basic CloudWatch logs; measure and log time-to-first-audio (target < 1.5s).

## Nova Sonic model (confirmed live)

- Region: **us-east-1** — `amazon.nova-2-sonic-v1:0` / `amazon.nova-2-5-sonic`.
- The avatar build and S3 assets live in **us-east-2** (SOW region); the Bedrock Nova
  Sonic call targets **us-east-1** where the model is available. Keep the two regions
  explicit in config.

## Status

Scaffold only. Implementation is Arc B Tasks 12–17 (audio→FLAME driver, Nova Sonic
session service, WebRTC transport, end-to-end integration, perf tiering, deploy/ops).
Nothing is installed yet; tooling choice is deferred to those tasks.
