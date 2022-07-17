import { Component } from '@angular/core';
import * as tf from '@tensorflow/tfjs'

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {
  title = 'tensorFlowApp';
  linearModel!: tf.Sequential
  prediction: any

  ngOnInit() {
    this.tranNewModel();
  }

  async trainNewModel() {
    this.linearModel = tf.sequential()

    this.linearModel.add(tf.layers.dense({units: 1, inputShape: [1]}))

    this.linearModel.compile({loss: 'meanSquareError', optimizer: 'sgd'})


    const xs = tf.tensorId([]);
    const ys = tf.tensorId([]);


  //Train
  await this.linearModel.fit(xs, ys);

  console.log('model trained')
  }

  linearPrediction(val) {
    const output = this.linearModel.predict(tf.tensor2d([val], [1,1])) as any;
    this.prediction = Array.from(output.datasync())[0]
  }

}