import { Injectable } from '@angular/core';
import { PtzService } from './ptz.service';
import { BehaviorSubject, Observable, Subject, concatWith, debounceTime, distinctUntilChanged, throttleTime, tap, delay } from 'rxjs';
import {bisectLeft, bisectRight} from 'd3-array'
import { RtcService } from './rtc.service';

@Injectable({
  providedIn: 'root'
})
export class GamepadService {

  constructor(private rtcService: RtcService) {
    this.scanGamepad();
    this.directionCommand$.pipe(
      distinctUntilChanged((prev, curr)=>{ return prev.x == curr.x && prev.y == curr.y && prev.spd == curr.spd;}),
      //throttleTime(200),
      tap((d)=>{console.log(d)}),
      tap((d)=>{this.rtcService.sendMessage(JSON.stringify({'T': 141, 'X': d.x, 'Y': d.y, 'SPD': d.spd}));}),
      //delay(200),
      //tap((d)=>{this.ptzService.sendMessage({'T': 2, 'X': 0, 'Y': 0, 'SPD': 0});})
    ).subscribe();
    this.resetDirectionCommand$.pipe(debounceTime(300)).subscribe(() => {
      this.rtcService.sendMessage(JSON.stringify({"T":133, "X":0, "Y":0, "SPD":0, "ACC":128}))
      // this.ptzService.sendMessage({"T":1,"X":0,"Y":0,"SPD":0,"ACC":128});
    });
    this.lightButtonCommand$.pipe(debounceTime(20)).subscribe(() => {
      if (this.isLightOn) {
        this.isLightOn = false;
        this.lightOn$.next(false);
      } else {
        this.isLightOn = true;
        this.lightOn$.next(true);
      }
    });
    this.lightOn.subscribe((on) => {
      this.rtcService.sendMessage(JSON.stringify({"T":132,"IO4": on ? 255 : 0,"IO5":on ? 255 : 0}))
    });
    this.moveCommand$.pipe(
      //debounceTime(300),
      throttleTime(50),
      distinctUntilChanged((prev, curr)=>{ return prev.x == curr.x && prev.z == curr.z}),
      tap((d)=>(this.rtcService.sendMessage(JSON.stringify({'T':13, 'X':d.x, 'Z':d.z}))))
    ).subscribe();
  }

  private yAxisSubject$ = new Subject<number>();
  private xAxisSubject$ = new Subject<number>();
  private xSpeed$ = new Subject<number>();
  private ySpeed$ = new Subject<number>();
  private speed$ = new Subject<number>();

  private lTrigger$ = new Subject<number>();
  private rTrigger$ = new Subject<number>();
  private lXAxis$ = new Subject<number>();

  yAxis: Observable<number> = this.yAxisSubject$.asObservable();
  xAxis: Observable<number> = this.xAxisSubject$.asObservable();

  xSpeed: Observable<number> = this.xSpeed$.asObservable();
  ySpeed: Observable<number> = this.ySpeed$.asObservable();
  speed: Observable<number> = this.speed$.asObservable();

  lTrigger = this.lTrigger$.asObservable();
  rTrigger = this.rTrigger$.asObservable();
  lXAxis = this.lXAxis$.asObservable();


  private directionCommand$ = new Subject<{ x: number, y: number, spd: number }>();
  private resetDirectionCommand$ = new Subject<void>();
  private moveCommand$ = new Subject<{x: number, z: number}>();

  private isLightOn = false;
  private lightOn$ = new BehaviorSubject<boolean>(false);
  lightOn: Observable<boolean> = this.lightOn$.asObservable();
  private lightButtonCommand$ = new Subject<void>();

  SPEED_AXIS_RANGE = [0.3, 0.5, 0.9, 1.1] // 0-0.3 => 0, 0.3-0.8 => 1, 0.8+ => 2
  SPEED_RANGE = [1, 10, 20, 50,100]


  scanGamepad = () => {

    const gamepads = navigator.getGamepads();
    if (gamepads) {
      // console.log(gamepads)
      const gp = gamepads[0];
      if (gp) {
        let speed_x = bisectRight(this.SPEED_AXIS_RANGE, Math.abs(gp.axes[2]));
        let speed_y = bisectRight(this.SPEED_AXIS_RANGE, Math.abs(gp.axes[3]));
        let spd = this.SPEED_RANGE[Math.max(speed_x, speed_y)];

        this.xAxisSubject$.next(gp.axes[2]);
        this.yAxisSubject$.next(gp.axes[3]);

        this.lXAxis$.next(gp.axes[0]);
        this.lTrigger$.next(gp.buttons[6].value);
        this.rTrigger$.next(gp.buttons[7].value);

        this.xSpeed$.next(speed_x);
        this.ySpeed$.next(speed_y);
        this.speed$.next(spd);

        let forwardSpeed = parseFloat((gp.buttons[6].value - gp.buttons[7].value).toFixed(1));
        let turningSpeed =  parseFloat((parseFloat(gp.axes[0].toFixed(1)) * -Math.PI).toFixed(1));
        
        

        if (speed_x !== 0 || speed_y !== 0) {
          this.directionCommand$.next({ x: Math.sign(gp.axes[2])*Math.min(speed_x,1), y: Math.sign(gp.axes[3])*-1*Math.min(speed_y,1), spd: spd });
        } else {
          this.directionCommand$.next({x: 0, y: 0, spd: 10});
        }
        if (gp.buttons[11].pressed) {
          this.resetDirectionCommand$.next();
        }
        if (gp.buttons[12].pressed) {
          this.lightButtonCommand$.next();
        }
        // if (forwardSpeed != 0 || Math.abs(turningSpeed) > 0.1) {
          this.moveCommand$.next({x: forwardSpeed, z: turningSpeed});
        // }
      }
    }

    requestAnimationFrame(this.scanGamepad)
  }

}
