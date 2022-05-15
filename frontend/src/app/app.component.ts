import {Component, OnInit} from '@angular/core';
import {DataService} from './data.service';
import {FormBuilder, FormControl, FormGroup, Validators} from '@angular/forms';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit {
  public form!: FormGroup;

  public submitted = false;

  constructor(private dataService: DataService,
              private formBuilder: FormBuilder) {}

  public ngOnInit(): void {
    this.form = this.formBuilder.group({
      startDate: new FormControl('', [Validators.required]),
      endDate: new FormControl('')
    });
  }

  public async onSubmit() {
    this.submitted = true;
    if (this.form.valid) {
      if (this.form.controls.endDate.value) {
        const values = await this.dataService.multiForecast(this.form.controls.startDate.value, this.form.controls.endDate.value);
      } else {
        const values = await this.dataService.simpleForecast(new Date(this.form.controls.startDate.value));
      }
    }
  }
}
