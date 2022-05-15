import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  constructor(private httpClient: HttpClient) { }

  public simpleForecast(date: Date): Array<number> {
    return [4, 7, 9];
/*
    return this.httpClient.post<string>('http:/localhost:5000/forecast', {date}).toPromise();
*/
  }

  public multiForecast(startDate: Date, endDate: Date): Array<Array<number>> {
    return [[4, 7, 9], [5, 6, 7]];
/*
    return this.httpClient.post<string>('http:/localhost:5000/forecast', { startDate, endDate }).toPromise();
*/
  }
}
