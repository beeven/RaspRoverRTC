import { Injectable } from '@angular/core';
import {io} from 'socket.io-client';
import { Observable, Subject, debounceTime, throttleTime } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class PtzService {
  // private socket = io()

  // private debouceSendMessageSubject = new Subject<string>();

  // constructor() {
  //   this.debouceSendMessageSubject.pipe(throttleTime(100)).subscribe((msg)=>{this.sendMessage(msg)});
  // }

  // sendMessage(message: any) {
  //   this.socket.emit('message', message);
  // }

  // getMessages() {
  //   let observable = new Observable<string>(observer => {
  //     this.socket.on('message', (data)=>{
  //       observer.next(data);
  //     });
  //     return ()=>{this.socket.disconnect();};
  //   });
  //   return observable;
  // }

  // sendCommand(cmd: any) {
  //   this.debouceSendMessageSubject.next(cmd);
  // }
}
