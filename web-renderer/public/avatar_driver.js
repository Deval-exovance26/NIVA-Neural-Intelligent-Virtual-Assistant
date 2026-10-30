// avatar_driver.js — FLAME-based animation driver for NIVA talking head avatar
// 
// Provides independent BLINK and LIP-SYNC drivers that modulate gaussians
// bound to a FLAME mesh. The FLAME mesh is posed per-frame, and bound gaussians
// follow the mesh deformation.
//
// REQUIREMENTS FROM TRACK A:
// - Each gaussian must have a parent FLAME triangle index
// - The export must include FLAME mesh topology (triangles, vertices)
// - The FLAME pose function must accept blendshape weights and jaw/eye params
//
// INPUT FORMAT (expected from loader):
// {
//   gaussians: { positions, scales, quats, opacity, sh_coeffs },
//   flame_mesh: { vertices: [V, 3], triangles: [T, 3] },
//   bindings: { triangle_index_per_gaussian: [N], barycentric_coords: [N, 3] }
// }
//
// ANIMATION PARAMETERS:
// - blink: FLAME eye-close blendshape (typically index in expression vector)
// - jaw: FLAME jaw rotation/translation param
// - eye: FLAME eye rotation params (left/right)

export class AvatarDriver {
  constructor(config = {}) {
    // Blink configuration
    this.blinkEnabled = config.blinkEnabled !== false;
    this.blinkInterval = config.blinkInterval || { min: 3000, max: 6000 }; // ms
    this.blinkDuration = config.blinkDuration || 120; // ms
    
    // Lip-sync configuration
    this.lipSyncEnabled = config.lipSyncEnabled !== false;
    this.jawOpenAmount = 0;
    this.jawSmoothing = config.jawSmoothing || 0.15;
    
    // Animation state
    this.blinkState = 0; // 0 = open, 1 = closed
    this.lastBlinkTime = performance.now();
    this.nextBlinkInterval = this._randomBlinkInterval();
    
    // FLAME params (will be populated after binding)
    this.flameParams = {
      expression: new Float32Array(100), // 100-dim expression code
      jaw: { rotation: 0, translation: [0, 0, 0] },
      eyes: { left: [0, 0, 0], right: [0, 0, 0] }
    };
    
    // Blendshape indices (FLAME standard)
    // These should match the FLAME model used during distillation
    this.BLENDSHAPE_EYE_CLOSE = config.eyeCloseIdx || 0; // Approximation
    this.BLENDSHAPE_JAW = config.jawIdx || 0; // First jaw-related blendshape
    
    // Timers
    this.blinkTimer = null;
    this.audioAnalyser = null;
    this.audioContext = null;
    this.audioSource = null;
  }
  
  /**
   * Generate random blink interval between min and max
   */
  _randomBlinkInterval() {
    const { min, max } = this.blinkInterval;
    return min + Math.random() * (max - min);
  }
  
  /**
   * Start the independent blink driver
   * Creates a timer that triggers blinks every 3-6 seconds
   */
  startBlinkDriver() {
    if (!this.blinkEnabled) return;
    
    this.lastBlinkTime = performance.now();
    this._scheduleNextBlink();
  }
  
  /**
   * Schedule the next blink
   */
  _scheduleNextBlink() {
    if (this.blinkTimer) {
      clearTimeout(this.blinkTimer);
    }
    
    this.nextBlinkInterval = this._randomBlinkInterval();
    this.blinkTimer = setTimeout(() => {
      this._executeBlink();
      this._scheduleNextBlink();
    }, this.nextBlinkInterval);
  }
  
  /**
   * Execute a single blink animation
   * Smoothly closes and opens eyes over ~120ms
   */
  _executeBlink() {
    const halfDuration = this.blinkDuration / 2;
    const startTime = performance.now();
    
    // Close phase
    const closeAnimation = () => {
      const elapsed = performance.now() - startTime;
      const t = Math.min(elapsed / halfDuration, 1);
      
      // Smooth ease-in-out for natural blink
      this.blinkState = this._easeInOutQuad(t);
      
      if (t < 1) {
        requestAnimationFrame(closeAnimation);
      } else {
        // Open phase (slightly faster for natural feel)
        const openStartTime = performance.now();
        const openAnimation = () => {
          const openElapsed = performance.now() - openStartTime;
          const openT = Math.min(openElapsed / (halfDuration * 0.7), 1);
          
          this.blinkState = 1 - this._easeInOutQuad(openT);
          
          if (openT < 1) {
            requestAnimationFrame(openAnimation);
          } else {
            this.blinkState = 0;
          }
        };
        requestAnimationFrame(openAnimation);
      }
    };
    
    requestAnimationFrame(closeAnimation);
  }
  
  /**
   * Ease-in-out quadratic function for smooth animation
   */
  _easeInOutQuad(t) {
    return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
  }
  
  /**
   * Stop the blink driver
   */
  stopBlinkDriver() {
    if (this.blinkTimer) {
      clearTimeout(this.blinkTimer);
      this.blinkTimer = null;
    }
    this.blinkState = 0;
  }
  
  /**
   * Initialize Web Audio API for lip-sync from audio file
   * @param {string} audioUrl - URL to audio file
   * @param {HTMLAudioElement} audioElement - Optional pre-created audio element
   */
  async initLipSyncFromAudio(audioUrl, audioElement = null) {
    try {
      this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
      
      // Create audio element if not provided
      this.audioElement = audioElement || new Audio();
      if (audioUrl) {
        this.audioElement.src = audioUrl;
        this.audioElement.crossOrigin = "anonymous";
      }
      
      // Create analyser node
      this.audioAnalyser = this.audioContext.createAnalyser();
      this.audioAnalyser.fftSize = 256;
      this.audioAnalyser.smoothingTimeConstant = this.jawSmoothing;
      
      // Connect audio source to analyser
      this.audioSource = this.audioContext.createMediaElementSource(this.audioElement);
      this.audioSource.connect(this.audioAnalyser);
      this.audioAnalyser.connect(this.audioContext.destination);
      
      // Frequency data buffer
      this.frequencyData = new Uint8Array(this.audioAnalyser.frequencyBinCount);
      
      return true;
    } catch (e) {
      console.warn("Failed to init audio lip-sync:", e);
      return false;
    }
  }
  
  /**
   * Start audio playback and lip-sync
   */
  async playAudio() {
    if (!this.audioElement) return;
    
    if (this.audioContext && this.audioContext.state === 'suspended') {
      await this.audioContext.resume();
    }
    
    await this.audioElement.play();
  }
  
  /**
   * Pause audio playback
   */
  pauseAudio() {
    if (this.audioElement) {
      this.audioElement.pause();
    }
  }
  
  /**
   * Update jaw position based on audio amplitude envelope
   * Should be called every frame from the render loop
   */
  updateJawFromAudio() {
    if (!this.audioAnalyser || !this.lipSyncEnabled) {
      this.jawOpenAmount *= 0.9; // Decay when no audio
      return;
    }
    
    // Get frequency data (0-255)
    this.audioAnalyser.getByteFrequencyData(this.frequencyData);
    
    // Compute amplitude envelope from low-mid frequencies (voice range)
    // Focus on frequencies 100-500 Hz (speech fundamental)
    const voiceRangeStart = Math.floor(100 * this.frequencyData.length / (this.audioContext.sampleRate / 2));
    const voiceRangeEnd = Math.floor(500 * this.frequencyData.length / (this.audioContext.sampleRate / 2));
    
    let amplitude = 0;
    for (let i = voiceRangeStart; i < voiceRangeEnd && i < this.frequencyData.length; i++) {
      amplitude += this.frequencyData[i];
    }
    amplitude /= (voiceRangeEnd - voiceRangeStart);
    
    // Normalize to 0-1 range
    this.jawOpenAmount = amplitude / 255;
    
    // Apply smoothing
    this.jawOpenAmount = this.jawOpenAmount * 0.7 + (this.jawOpenAmount || 0) * 0.3;
  }
  
  /**
   * Set jaw position manually (for external lip-sync driver)
   * @param {number} amount - Jaw open amount [0, 1]
   */
  setJawAmount(amount) {
    this.jawOpenAmount = Math.max(0, Math.min(1, amount));
  }
  
  /**
   * Set expression code directly (for external animation driver)
   * @param {Float32Array} expressionCode - 100-dim FLAME expression code
   */
  setExpressionCode(expressionCode) {
    if (expressionCode.length === 100) {
      this.flameParams.expression = expressionCode;
    }
  }
  
  /**
   * Get current FLAME parameters for rendering
   * Combines blink state, jaw position, and any external expression code
   */
  getFlameParams() {
    // Start with base expression code
    const expression = new Float32Array(this.flameParams.expression);
    
    // Apply blink to expression (modulate eye-close blendshape)
    // In FLAME, eye blink is typically a combination of blendshapes
    // We approximate with a single blendshape index
    expression[this.BLENDSHAPE_EYE_CLOSE] = this.blinkState;
    
    // Apply jaw open (rotation in radians, scaled to natural range)
    const jawRotation = this.jawOpenAmount * 0.3; // Max ~17 degrees
    
    return {
      expression,
      jaw: {
        rotation: jawRotation,
        translation: [0, this.jawOpenAmount * 0.02, 0] // Slight forward translation
      },
      eyes: {
        left: this.flameParams.eyes.left,
        right: this.flameParams.eyes.right
      }
    };
  }
  
  /**
   * Update animation state (call every frame)
   */
  update() {
    // Update jaw from audio if available
    this.updateJawFromAudio();
  }
  
  /**
   * Clean up resources
   */
  dispose() {
    this.stopBlinkDriver();
    
    if (this.audioContext) {
      this.audioContext.close();
      this.audioContext = null;
    }
    
    if (this.audioElement) {
      this.audioElement.pause();
      this.audioElement.src = "";
      this.audioElement = null;
    }
  }
}

/**
 * FLAME mesh poser utility
 * Deforms mesh vertices based on expression parameters
 * 
 * This is a simplified implementation. For production, use the
 * full FLAME model with learned blendshapes.
 */
export class FlamePoser {
  constructor(flameMesh, config = {}) {
    this.mesh = flameMesh;
    this.numVertices = flameMesh.vertices.length / 3;
    
    // Blendshape basis (should be loaded from FLAME model)
    // For now, we use a simple approximation
    this.expressionBasis = config.expressionBasis || null;
    this.jawBasis = config.jawBasis || null;
    
    // Cache for deformed vertices
    this.deformedVertices = new Float32Array(flameMesh.vertices);
  }
  
  /**
   * Pose the FLAME mesh with given parameters
   * @param {Object} params - { expression, jaw, eyes }
   * @returns {Float32Array} Deformed vertex positions
   */
  pose(params) {
    const { expression, jaw, eyes } = params;
    
    // Start with base mesh
    const baseVertices = this.mesh.vertices;
    const deformed = this.deformedVertices;
    
    // Copy base vertices
    deformed.set(baseVertices);
    
    // Apply expression blendshapes (if basis available)
    if (this.expressionBasis && expression) {
      this._applyBlendshapes(deformed, expression, this.expressionBasis);
    }
    
    // Apply jaw rotation/translation
    if (jaw) {
      this._applyJaw(deformed, jaw);
    }
    
    // Apply eye rotations (for gaze direction)
    if (eyes) {
      this._applyEyes(deformed, eyes);
    }
    
    return deformed;
  }
  
  /**
   * Apply blendshape deformations
   * Simplified - assumes blendshape basis is [V, 100, 3]
   */
  _applyBlendshapes(vertices, expression, basis) {
    // This is a placeholder - real implementation needs the FLAME basis
    // For now, we just return the vertices unchanged
    // The actual deformation should be done in the shader or with WASM
    // for performance
  }
  
  /**
   * Apply jaw deformation
   * Rotates vertices in the jaw region around the jaw joint
   */
  _applyJaw(vertices, jaw) {
    const { rotation, translation } = jaw;
    
    // Jaw joint location (approximate - should come from FLAME model)
    const jawJoint = [0, -0.05, 0.08]; // Below ear, behind chin
    
    // Jaw vertex indices (approximate - should come from FLAME segmentation)
    // This is a placeholder for demonstration
    const rotationMatrix = this._rotationMatrixX(rotation);
    
    // Apply to lower face vertices (simplified)
    // In production, this would use proper vertex masks from FLAME
    for (let i = 0; i < this.numVertices; i++) {
      const idx = i * 3;
      const y = baseVertices[idx + 1];
      
      // Apply to vertices below jaw joint (lower face/jaw)
      if (y < jawJoint[1] + 0.02) {
        // Translate to jaw joint
        const x = vertices[idx] - jawJoint[0];
        const yLocal = vertices[idx + 1] - jawJoint[1];
        const z = vertices[idx + 2] - jawJoint[2];
        
        // Rotate
        vertices[idx] = jawJoint[0] + x;
        vertices[idx + 1] = jawJoint[1] + yLocal * Math.cos(rotation) - z * Math.sin(rotation);
        vertices[idx + 2] = jawJoint[2] + yLocal * Math.sin(rotation) + z * Math.cos(rotation);
        
        // Apply translation
        vertices[idx] += translation[0];
        vertices[idx + 1] += translation[1];
        vertices[idx + 2] += translation[2];
      }
    }
  }
  
  /**
   * Apply eye rotations (gaze direction)
   */
  _applyEyes(vertices, eyes) {
    // Placeholder for eye rotation
    // In FLAME, eyes have separate rotation parameters
    // This would deform vertices in the eye region
  }
  
  /**
   * Create rotation matrix around X axis
   */
  _rotationMatrixX(angle) {
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    return [
      1, 0, 0,
      0, c, -s,
      0, s, c
    ];
  }
}

/**
 * Gaussian binding utility
 * Computes per-gaussian deformation based on parent triangle deformation
 */
export class GaussianBinding {
  constructor(gaussians, flameMesh, bindings) {
    this.gaussians = gaussians;
    this.mesh = flameMesh;
    this.bindings = bindings;
    
    // Cache for deformed gaussian positions
    this.deformedPositions = new Float32Array(gaussians.positions);
  }
  
  /**
   * Update gaussian positions based on deformed FLAME mesh
   * @param {Float32Array} deformedVertices - Deformed FLAME vertex positions
   * @returns {Float32Array} Deformed gaussian positions
   */
  updatePositions(deformedVertices) {
    const { triangle_index_per_gaussian, barycentric_coords } = this.bindings;
    const basePositions = this.gaussians.positions;
    const deformed = this.deformedPositions;
    
    const numGaussians = triangle_index_per_gaussian.length;
    
    for (let i = 0; i < numGaussians; i++) {
      const triIdx = triangle_index_per_gaussian[i];
      const bary = barycentric_coords[i];
      
      // Get triangle vertex indices
      const v0Idx = this.mesh.triangles[triIdx * 3 + 0];
      const v1Idx = this.mesh.triangles[triIdx * 3 + 1];
      const v2Idx = this.mesh.triangles[triIdx * 3 + 2];
      
      // Get deformed vertex positions
      const v0 = [
        deformedVertices[v0Idx * 3 + 0],
        deformedVertices[v0Idx * 3 + 1],
        deformedVertices[v0Idx * 3 + 2]
      ];
      const v1 = [
        deformedVertices[v1Idx * 3 + 0],
        deformedVertices[v1Idx * 3 + 1],
        deformedVertices[v1Idx * 3 + 2]
      ];
      const v2 = [
        deformedVertices[v2Idx * 3 + 0],
        deformedVertices[v2Idx * 3 + 1],
        deformedVertices[v2Idx * 3 + 2]
      ];
      
      // Compute deformed position via barycentric interpolation
      const baseIdx = i * 3;
      deformed[baseIdx + 0] = v0[0] * bary[0] + v1[0] * bary[1] + v2[0] * bary[2];
      deformed[baseIdx + 1] = v0[1] * bary[0] + v1[1] * bary[1] + v2[1] * bary[2];
      deformed[baseIdx + 2] = v0[2] * bary[0] + v1[2] * bary[1] + v2[2] * bary[2];
    }
    
    return deformed;
  }
  
  /**
   * Compute per-gaussian rotation based on triangle normal change
   * This helps gaussians follow the surface deformation more naturally
   */
  updateRotations(deformedVertices, baseRotations) {
    // Placeholder for rotation update
    // This would compute the rotation difference between base and deformed
    // triangle normals and apply it to gaussian rotations
    return baseRotations;
  }
}
