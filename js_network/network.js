var brain = require('brain.js');


var fs=require('fs');
var data_tairn=fs.readFileSync('train.json', 'utf8');
var train=JSON.parse(data_tairn);
// console.log(words);

const config = {
    hiddenLayers: [ 50, 50], 
    activation:  'sigmoid',
    // praxis: 'adam'
    // leakyReluAlpha:  0.01
   }

var net = new brain.NeuralNetwork();

net.train(train,{
    // errorThresh: 0.005,  // error threshold to reach
    iterations: 5000,   // maximum training iterations
    log: true,           // console.log() progress periodically
    logPeriod: 100,       // number of iterations between logging
    learningRate: 0.001, 
    praxis: 'adam'  // learning rate
});

let wstream = fs.createWriteStream('model.json');
wstream.write(JSON.stringify(net.toJSON(),null,2));
wstream.end();

var data_test = fs.readFileSync('test.json', 'utf8');
var test = JSON.parse(data_test);
let count = 0;
console.log(net.run(test[0].input))
for (let i=0; i<test.length; i++){
    
    if (net.run(test[i].input)>0.5){
        if (test[i].output[0] == 1){
            count++
        }
    }else{
        if (test[i].output[0] == 0){
            count++
        }
    }
}
console.log(count/test.length);
// console.log(net.run([1, 0, 1, 0, 1, 0, 1, 1.8936170212765957, 18.333333333333332, 0, 667]))


