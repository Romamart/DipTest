let brain = require('brain.js');


let fs=require('fs');
let max_acc_coef = JSON.parse(fs.readFileSync('Data/max_acc.json', 'utf8')) + 1;
console.log(max_acc_coef);
let data_tairn=fs.readFileSync(`Data/train${max_acc_coef}.json`, 'utf8');
let train = JSON.parse(data_tairn);
let epochs = JSON.parse(fs.readFileSync('Data/epochs.json', 'utf8'));
const config = {
    hiddenLayers: [10], 
    activation:  'sigmoid',
   }

let net = new brain.NeuralNetwork(config);
console.log(epochs[max_acc_coef])
net.train(train,{
    errorThresh: 0.005,
    iterations: epochs[max_acc_coef], 
    log: true,          
    logPeriod: 100,      
    learningRate: 0.001, 
    praxis: 'adam'
});

let wstream = fs.createWriteStream('model.json');
wstream.write(JSON.stringify(net.toJSON(),null,2));
wstream.end();

let data_test = fs.readFileSync(`Data/test${max_acc_coef}.json`, 'utf8');
let test = JSON.parse(data_test);
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
let data_test_gen = fs.readFileSync(`test.json`, 'utf8');
let test_gen = JSON.parse(data_test_gen);
let count_gen = 0;
console.log(net.run(test_gen[0].input))
for (let i=0; i<test_gen.length; i++){
    
    if (net.run(test_gen[i].input)>0.5){
        if (test_gen[i].output[0] == 1){
            count_gen++
        }
    }else{
        if (test_gen[i].output[0] == 0){
            count_gen++
        }
    }
}
console.log(count/test.length);
console.log(count_gen/test_gen.length);
console.log(net.run([1, 0, 1, 1, 0, 1, 1.8936170212765957, 18.333333333333332, 0, 667]))


