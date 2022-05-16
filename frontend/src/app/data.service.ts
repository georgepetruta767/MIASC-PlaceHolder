import { Injectable } from '@angular/core';
import {HttpClient} from '@angular/common/http';
import {ForecastData} from './forecast-data';

@Injectable({
  providedIn: 'root'
})
export class DataService {

  constructor(private httpClient: HttpClient) { }

  public simpleForecast(date: Date): Promise<ForecastData> {
    const formData = new FormData();
    formData.append('date', reformatDate(date.toString()));
    return this.httpClient.post<ForecastData>('http://localhost:5000/forecast', formData).toPromise();
  }

  public multiForecast(startDate: Date, endDate: Date): Promise<ForecastData> {
    const formData = new FormData();
    formData.append('beginDate', reformatDate(startDate.toString()));
    formData.append('endDate', reformatDate(endDate.toString()));
    return this.httpClient.post<ForecastData>('http://localhost:5000/multiForecast', formData).toPromise();
  }
}

function reformatDate(date: string): string {
  return date.replace('-', '/').replace('-', '/');
}
