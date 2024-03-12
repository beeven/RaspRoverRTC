import { Component, OnInit } from '@angular/core';
import { RouterOutlet } from '@angular/router';
import { RtcService } from './rtc.service';
import { FormControl, ReactiveFormsModule } from '@angular/forms';
import { GamepadService } from './gamepad.service';
import { Observable } from 'rxjs';
import { AsyncPipe } from '@angular/common';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [RouterOutlet, ReactiveFormsModule, AsyncPipe],
  templateUrl: './app.component.html',
  styleUrl: './app.component.less'
})
export class AppComponent implements OnInit {
  title = 'ptzconsole';

  useStun = new FormControl(false);
  messages: string[] = [];
  xAxis: Observable<number>;
  yAxis: Observable<number>;
  lightOn: Observable<boolean>;
  xSpeed: Observable<number>;
  ySpeed: Observable<number>;
  speed: Observable<number>;

  lTrigger: Observable<number>;
  rTrigger: Observable<number>;
  lXAxis: Observable<number>;

  rtcStarted = false;
  videoStreams: MediaStream[] = [];
  audioStreams: MediaStream[] = [];
  offer: string | undefined;
  answer: string | undefined;
  connectionState: string = '';

  message = new FormControl('');

  constructor(private gamepadService: GamepadService, private rtcService: RtcService) {
    this.xAxis = gamepadService.xAxis;
    this.yAxis = gamepadService.yAxis;
    this.lightOn = gamepadService.lightOn;
    this.xSpeed = gamepadService.xSpeed;
    this.ySpeed = gamepadService.ySpeed;
    this.speed = gamepadService.speed;

    this.lXAxis = gamepadService.lXAxis;
    this.lTrigger = gamepadService.lTrigger;
    this.rTrigger = gamepadService.rTrigger;

    this.rtcService.audioStream.subscribe(stream => { this.audioStreams.push(stream); });
    this.rtcService.videoStream.subscribe(stream =>{ this.videoStreams.push(stream); });
    this.rtcService.offerSdp.subscribe(content =>{ this.offer = content;});
    this.rtcService.answerSdp.subscribe(content => {this.answer = content;});
    this.rtcService.connectionState.subscribe(state => {this.connectionState = state});
  }
  ngOnInit(): void {
    // this.ptzService.getMessages().subscribe((message: string)=>{
    //   this.messages.push(message);
    //   if(this.messages.length > 10) {
    //     this.messages.shift();
    //   }
    // })
    this.rtcService.getMessages().subscribe(msg => {
      this.messages.push(msg);
      if(this.messages.length > 10) {
        this.messages.shift();
      }
    });
  }

  async toggleRTC() {
    if (this.rtcStarted) {
      await this.rtcService.stop();
      this.videoStreams = [];
      this.audioStreams = [];
      this.rtcStarted = false;
    } else {
      this.rtcStarted = true;
      await this.rtcService.start(this.useStun.value);
    }
  }

  sendMsg() {
    this.rtcService.sendMessage(this.message.value);
    this.message.setValue('');
  }

}
