/**
 * One racer's car: animates server ``drive`` events on the race clock and shows
 * thinking / finished states. Positions are derived from event timestamps, so a
 * browser that reconnects mid-race snaps every car to the right place.
 */

import * as THREE from 'three';
import { CAR } from '../config.js';
import { angleDelta, easeInOutQuad, easeOutCubic, headingToAngle, junctionToWorld } from '../geometry.js';
import { buildCarModel } from './car-model.js';

const TAIL_IDLE = new THREE.Color(0x7a1010);
const TAIL_BRAKE = new THREE.Color(0xff2a2a);

export class Car {
  /**
   * @param {object} racer Racer view from ``race_init`` (id, name, device, color, heading).
   * @param {[number, number]} startNode Start junction.
   * @param {number} laneOffset Sideways offset so cars on the same road don't overlap.
   */
  constructor(racer, startNode, laneOffset) {
    this.racer = racer;
    this.laneOffset = laneOffset;
    this.object = buildCarModel({ color: racer.color, plate: racer.label });
    this.parts = this.object.userData.parts;

    this.base = junctionToWorld(startNode); // junction-center position without lane offset
    this.yaw = headingToAngle(racer.heading);
    this.drive = null;
    this.thinking = false;
    this.finished = false;
    this.speed = 0; // world units/sec, used for engine sound and camera feel
    this._lastBase = this.base.clone();
    this.applyTransform(0);
  }

  /** Forward direction (unit vector) the car is facing. */
  get forward() {
    return new THREE.Vector3(-Math.sin(this.yaw), 0, -Math.cos(this.yaw));
  }

  /** @param {boolean} value */
  setThinking(value) {
    this.thinking = value;
  }

  setFinished() {
    this.finished = true;
    this.thinking = false;
  }

  /**
   * Start animating a road segment.
   * @param {{src: number[], dst: number[], heading: string, duration_s: number, t: number}} event
   * @param {boolean} isUturn
   */
  startDrive(event, isUturn) {
    this.finishCurrentDrive();
    this.thinking = false;
    this.drive = {
      from: junctionToWorld(event.src),
      to: junctionToWorld(event.dst),
      fromYaw: this.yaw,
      toYaw: headingToAngle(event.heading),
      start: event.t,
      duration: event.duration_s,
      uturn: isUturn,
    };
  }

  /** Jump to the end of the current segment (used when a new one arrives early). */
  finishCurrentDrive() {
    if (!this.drive) return;
    this.base.copy(this.drive.to);
    this.yaw = this.drive.toYaw;
    this.drive = null;
  }

  /**
   * Advance the animation.
   * @param {number} raceTime Current race clock in seconds (server time base).
   * @param {number} dt Frame delta in seconds.
   * @param {number} wallTime Free-running time for idle effects.
   */
  update(raceTime, dt, wallTime) {
    if (this.drive) this.updateDrive(raceTime);
    if (dt > 0) {
      const moved = this.base.distanceTo(this._lastBase);
      this.speed = THREE.MathUtils.lerp(this.speed, moved / dt, 1 - Math.exp(-dt * 8));
    }
    this._lastBase.copy(this.base);
    this.applyTransform(wallTime);
  }

  updateDrive(raceTime) {
    const drive = this.drive;
    const p = THREE.MathUtils.clamp((raceTime - drive.start) / drive.duration, 0, 1);
    const turnPortion = drive.uturn ? CAR.uturnPortion : CAR.turnPortion;
    const turn = easeOutCubic(Math.min(1, p / turnPortion));
    this.yaw = drive.fromYaw + angleDelta(drive.fromYaw, drive.toYaw) * turn;

    // U-turns pivot in place first; normal turns accelerate out of the junction and
    // brake into the next one (ease in/out), which reads as "slowing at junctions".
    const travel = drive.uturn ? Math.max(0, (p - turnPortion) / (1 - turnPortion)) : p;
    this.base.lerpVectors(drive.from, drive.to, easeInOutQuad(travel));
    if (p >= 1) this.drive = null;
  }

  applyTransform(wallTime) {
    const right = new THREE.Vector3(Math.cos(this.yaw), 0, -Math.sin(this.yaw));
    this.object.position.copy(this.base).addScaledVector(right, this.laneOffset);
    this.object.rotation.y = this.yaw;

    // Idle engine shake while waiting for the model, brake lights on when stopped.
    const idle = this.speed < 0.5 && !this.finished;
    this.object.position.y = idle ? Math.abs(Math.sin(wallTime * 38)) * 0.03 : 0;
    const braking = this.thinking || (this.drive && this.speed < 4);
    for (const tail of this.parts.tailLights) tail.material.color.copy(braking ? TAIL_BRAKE : TAIL_IDLE);

    if (this.parts.beacon) {
      const pulse = this.thinking ? 0.6 + Math.abs(Math.sin(wallTime * 9)) * 0.9 : 1;
      this.parts.beacon.scale.setScalar(pulse);
      this.parts.beacon.visible = this.thinking || this.finished || Math.sin(wallTime * 2) > -0.2;
    }
  }
}
