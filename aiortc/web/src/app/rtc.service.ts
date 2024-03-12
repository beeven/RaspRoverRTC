import { Injectable } from '@angular/core';
import { Observable, Subject, fromEvent } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RtcService {


  private videoStream$ = new Subject<MediaStream>();
  private audioStream$ = new Subject<MediaStream>();
  public videoStream = this.videoStream$.asObservable();
  public audioStream = this.audioStream$.asObservable();

  private pc: RTCPeerConnection | null = null;
  private localStream: MediaStream | null = null;
  private channel: RTCDataChannel | null = null;
  private channelData$ = new Subject<any>();

  
  private offerSdp$ = new Subject<string>();
  private answerSdp$ = new Subject<string>();
  public offerSdp = this.offerSdp$.asObservable();
  public answerSdp = this.answerSdp$.asObservable();

  private connectionState$ = new Subject<string>();
  public connectionState = this.connectionState$.asObservable();

  constructor() { }

  public async start(useStun: boolean | null = false) {
    if (this.pc != null) return;

    let config: RTCConfiguration = {};
    if (useStun) {
      config.iceServers = [{ urls: ['stun:stun.l.google.com:19302'] }]
    }

    this.pc = new RTCPeerConnection(config);

    this.connectionState$.next(this.pc.connectionState);
    this.pc.onconnectionstatechange = ev => {
      this.connectionState$.next(this.pc!.connectionState);
    }
    
    this.channel = this.pc.createDataChannel('command');
    this.channel.onopen = ev => {
      console.log("data channel opened.")
    }
    this.channel.onclose = ev => {
      console.log("data channel closed.")
    }
    this.channel.onerror = ev => {
      console.log("data channl error.")
    }
    this.channel.onmessage = ev => {
      this.channelData$.next(ev.data);
    }
    

    this.pc.addEventListener('track', event => {
      if (event.track.kind == 'video') {
        this.videoStream$.next(new MediaStream([event.track]));
      } else {
        this.audioStream$.next(new MediaStream([event.track]));
      }
    });

    this.localStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

    for (const track of this.localStream.getAudioTracks()) {
      this.pc.addTrack(track);
    }

    this.pc.addTransceiver('video', { direction: 'recvonly' });
    this.pc.addTransceiver('video', {direction: 'recvonly'});
    this.pc.addTransceiver('audio', { direction: 'sendrecv' });

    let offer = await this.pc.createOffer();
    await this.pc.setLocalDescription(offer);

    await (new Promise(resolve => {
      if (this.pc!.iceGatheringState === 'complete') { resolve(null); }
      else { this.pc!.addEventListener('icegatheringstatechange', () => this.pc!.iceGatheringState === 'complete' && resolve(null)) }
    }));

    this.offerSdp$.next(this.pc.localDescription?.sdp ?? '');

    let resp = await fetch('/offer', {
      body: JSON.stringify({
        sdp: offer.sdp,
        type: offer.type,
      }),
      headers: {
        'Content-Type': 'application/json'
      },
      method: 'POST'
    });

    let answer = await resp.json();
    this.answerSdp$.next(answer.sdp);
    await this.pc.setRemoteDescription(answer);
    
  }

  public async stop() {

    if (this.pc != null) {
      this.channel?.close();
      this.pc.close();
      this.pc = null;
      this.offerSdp$.next('');
      this.answerSdp$.next('');
      this.connectionState$.next('');
    }
    if (this.localStream != null) {
      this.localStream.getTracks().forEach(track => {
        track.stop();
      });
      this.localStream = null;
    }
  }

  public sendMessage(message: any ) {
    if(this.channel && this.channel.readyState === "open") {
      console.log("sending message:", message);
      this.channel.send(message);
    } else {
      console.log("data channel not ready:", this.channel?.readyState);
    }
  }
  public getMessages() {
    return this.channelData$.asObservable();
  }
}
