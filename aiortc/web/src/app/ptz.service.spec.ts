import { TestBed } from '@angular/core/testing';

import { PtzService } from './ptz.service';

describe('PtzService', () => {
  let service: PtzService;

  beforeEach(() => {
    TestBed.configureTestingModule({});
    service = TestBed.inject(PtzService);
  });

  it('should be created', () => {
    expect(service).toBeTruthy();
  });
});
