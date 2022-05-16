import {DataService} from './data.service';
import { EChartsOption } from 'echarts';
import {Component, OnInit} from '@angular/core';
import {FormBuilder, FormControl, FormGroup, Validators} from '@angular/forms';
import {ForecastData} from './forecast-data';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  public form!: FormGroup;

  public submitted = false;
  public chartOption: EChartsOption = {
    xAxis: {
      type: 'category',
      data: [],
    },
    yAxis: {
      type: 'value',
    },
    series: [
      {
        data: [],
        type: 'line',
      },
    ],
  };

  constructor(private dataService: DataService,
              private formBuilder: FormBuilder) {}

  public ngOnInit(): void {
    this.form = this.formBuilder.group({
      startDate: new FormControl('', [Validators.required]),
      endDate: new FormControl('')
    });
  }

  public async onSubmit(): Promise<void> {
    this.submitted = true;
    if (this.form.valid) {
      let values: ForecastData;
      if (this.form.controls.endDate.value) {
        values = await this.dataService.multiForecast(this.form.controls.startDate.value, this.form.controls.endDate.value);
      } else {
        values = await this.dataService.simpleForecast(new Date(this.form.controls.startDate.value));
      }

      this.chartOption = {
        xAxis: {
          type: 'category',
          data: values.date,
        },
        yAxis: {
          type: 'value',
        },
        series: [
          {
            data: values.med,
            type: 'line',
          },
          {
            data: values.max,
            type: 'line',
          },
          {
            data: values.min,
            type: 'line',
          },
        ],
      };
    }
  }
}
