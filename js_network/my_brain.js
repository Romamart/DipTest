class lookup {

    static toTable(hashes) {
        const hash = hashes.reduce((memo, hash) => {
        return Object.assign(memo, hash);
        }, {});
  
        return lookup.toHash(hash);
    }
  
    
    static toTable2D(objects2D) {
        const table = {};
        let valueIndex = 0;
        for (let i = 0; i < objects2D.length; i++) {
            const objects = objects2D[i];
            for (let j = 0; j < objects.length; j++) {
                const object = objects[j];
                for (const p in object) {
                    if (object.hasOwnProperty(p) && !table.hasOwnProperty(p)) {
                        table[p] = valueIndex++;
                    }
                }
            }
        }
        return table;
    }
  
    static toInputTable(data) {
        const table = {};
        let tableIndex = 0;
        for (let dataIndex = 0; dataIndex < data.length; dataIndex++) {
            for (let p in data[dataIndex].input) {
                if (!table.hasOwnProperty(p)) {
                    table[p] = tableIndex++;
                }
            }
        }
        return table;
    }
  
    static toInputTable2D(data) {
        const table = {};
        let tableIndex = 0;
        for (let dataIndex = 0; dataIndex < data.length; dataIndex++) {
            const input = data[dataIndex].input;
            for (let i = 0; i < input.length; i++) {
                const object = input[i];
                for (let p in object) {
                    if (!table.hasOwnProperty(p)) {
                        table[p] = tableIndex++;
                    }
                }
            }
        }
        return table;
    }
  
    static toOutputTable(data) {
        const table = {};
        let tableIndex = 0;
        for (let dataIndex = 0; dataIndex < data.length; dataIndex++) {
            for (let p in data[dataIndex].output) {
                if (!table.hasOwnProperty(p)) {
                    table[p] = tableIndex++;
                }
            }
        }
        return table;
    }
  
    static toOutputTable2D(data) {
        const table = {};
        let tableIndex = 0;
        for (let dataIndex = 0; dataIndex < data.length; dataIndex++) {
            const output = data[dataIndex].output;
            for (let i = 0; i < output.length; i++) {
                const object = output[i];
                for (let p in object) {
                    if (!table.hasOwnProperty(p)) {
                        table[p] = tableIndex++;
                    }
                }
            }
        }
        return table;
    }
  
    
    static toHash(hash) {
        let lookup = {};
        let index = 0;
        for (let i in hash) {
            lookup[i] = index++;
        }
        return lookup;
    }
  
    
    static toArray(lookup, object, arrayLength) {
        const result = new Float32Array(arrayLength);
        for (let p in lookup) {
            result[lookup[p]] = object.hasOwnProperty(p) ? object[p] : 0;
        }
        return result;
    }
  
    static toArrayShort(lookup, object) {
        const result = [];
        for (let p in lookup) {
            if (!object.hasOwnProperty(p)) break;
            result[lookup[p]] = object[p];
        }
        return Float32Array.from(result);
    }
  
    static toArrays(lookup, objects, arrayLength) {
        const result = [];
        for (let i = 0; i < objects.length; i++) {
            result.push(this.toArray(lookup, objects[i], arrayLength));
        }
        return result;
    }
  
    static toObject(lookup, array) {
        const object = {};
        for (let p in lookup) {
            object[p] = array[lookup[p]];
        }
        return object;
    }
  
    static toObjectPartial(lookup, array, offset = 0, limit = 0) {
        const object = {};
        let i = 0;
        for (let p in lookup) {
            if (offset > 0) {
                if (i++ < offset) continue;
            }
            if (limit > 0) {
                if (i++ >= limit) continue;
            }
            object[p] = array[lookup[p] - offset];
        }
        return object;
    }
  
    static lookupFromArray(array) {
        let lookup = {};
        let z = 0;
        let i = array.length;
        while (i-- > 0) {
            lookup[array[i]] = z++;
        }
        return lookup;
    }
  
    static dataShape(data) {
        const shape = [];
  
        if (data.input) {
            shape.push('datum');
            data = data.input;
        } else if (Array.isArray(data)) {
            if (data[0].input) {
                shape.push('array', 'datum');
                data = data[0].input;
            } else {
                shape.push('array');
                data = data[0];
            }
        }
  
        let p;
        while (data) {
            for (p in data) { break; }
            if (!data.hasOwnProperty(p)) break;
            if (Array.isArray(data) || data.buffer instanceof ArrayBuffer) {
                shape.push('array');
                data = data[p];
            } else if (typeof data === 'object') {
                shape.push('object');
                data = data[p];
            } else {
                throw new Error('unhandled signature');
            }
        }
        shape.push(typeof data);
        return shape;
    }
  
    static addKeys(value, table) {
        if (Array.isArray(value)) return;
        table = table || {};
        let i = Object.keys(table).length;
        for (const p in value) {
            if (!value.hasOwnProperty(p)) continue;
            if (table.hasOwnProperty(p)) continue;
            table[p] = i++;
        }
        return table;
    }
}

let toArray = function(values) {
    if (Array.isArray(values)) {
        return values;
    }
    return new Float32Array(Object.values(values));
};

let zeros = function(size) {
    return new Float32Array(size);
};

let randomWeight = function() {
    return Math.random() * 0.4 - 0.2;
}

let randos = function(size) {
    const array = new Float32Array(size);
    for (let i = 0; i < size; i++) {
      array[i] = randomWeight();
    }
    return array;
  };


class NeuralNetwork {
    static get trainDefaults() {
      return {
        iterations: 20000,    // the maximum times to iterate the training data
        errorThresh: 0.005,   // the acceptable error percentage from training data
        log: false,           // true to use console.log, when a function is supplied it is used
        logPeriod: 10,        // iterations between logging out
        learningRate: 0.3,    // multiply's against the input and the delta then adds to momentum
        momentum: 0.1,        // multiply's against the specified "change" then adds to learning rate for change
        callback: null,       // a periodic call back that can be triggered while training
        callbackPeriod: 10,   // the number of iterations through the training data between callback calls
        timeout: Infinity,    // the max number of milliseconds to train for
        praxis: null,
        beta1: 0.9,
        beta2: 0.999,
        epsilon: 1e-8,
      };
    }
  
    static get defaults() {
      return {
        leakyReluAlpha: 0.01,
        binaryThresh: 0.5,
        hiddenLayers: null,     // array of ints for the sizes of the hidden layers in the network
        activation: 'sigmoid'  // Supported activation types ['sigmoid', 'relu', 'leaky-relu', 'tanh']
      };
    }
  
    constructor(options = {}) {
        Object.assign(this, this.constructor.defaults, options);
        this.trainOpts = {};
        this.updateTrainingOptions(Object.assign({}, this.constructor.trainDefaults, options));
  
        this.sizes = null;
        this.outputLayer = null;
        this.biases = null; // weights for bias nodes
        this.weights = null;
        this.outputs = null;
    
        // state for training
        this.deltas = null;
        this.changes = null; // for momentum
        this.errors = null;
        this.errorCheckInterval = 1;
        if (!this.constructor.prototype.hasOwnProperty('runInput')) {
            this.runInput = null;
        }
        if (!this.constructor.prototype.hasOwnProperty('calculateDeltas')) {
            this.calculateDeltas = null;
        }
        this.inputLookup = null;
        this.inputLookupLength = null;
        this.outputLookup = null;
        this.outputLookupLength = null;
    
        if (options.inputSize && options.hiddenLayers && options.outputSize) {
            this.sizes = [options.inputSize]
            .concat(options.hiddenLayers)
            .concat([options.outputSize]);
            }
    }

    _runInputSigmoid(input) {
        this.outputs[0] = input;  // set output state of input layer
    
        let output = null;
        for (let layer = 1; layer <= this.outputLayer; layer++) {
            const activeLayer = this.sizes[layer];
            const activeWeights = this.weights[layer];
            const activeBiases = this.biases[layer];
            const activeOutputs = this.outputs[layer];
            for (let node = 0; node < activeLayer; node++) {
                let weights = activeWeights[node];
        
                let sum = activeBiases[node];
                for (let k = 0; k < weights.length; k++) {
                    sum += weights[k] * input[k];
                }
                //sigmoid
                activeOutputs[node] = 1 / (1 + Math.exp(-sum));
            }
            output = input = this.outputs[layer];
        }
        return output;
    }
    
    setActivation(activation) {
        this.activation = activation ? activation : this.activation;
        switch (this.activation) {
            case 'sigmoid':
                this.runInput = this.runInput || this._runInputSigmoid;
                this.calculateDeltas = this.calculateDeltas || this._calculateDeltasSigmoid;
                break;
            case 'relu':
                this.runInput = this.runInput || this._runInputRelu;
                this.calculateDeltas = this.calculateDeltas || this._calculateDeltasRelu;
                break;
            case 'leaky-relu':
                this.runInput = this.runInput || this._runInputLeakyRelu;
                this.calculateDeltas = this.calculateDeltas || this._calculateDeltasLeakyRelu;
                break;
            case 'tanh':
                this.runInput = this.runInput || this._runInputTanh;
                this.calculateDeltas = this.calculateDeltas || this._calculateDeltasTanh;
                break;
            default:
                throw new Error('unknown activation ' + this.activation + ', The activation should be one of [\'sigmoid\', \'relu\', \'leaky-relu\', \'tanh\']');
        }
    }

    logTrainingStatus(status) {
        console.log(`iterations: ${status.iterations}, training error: ${status.error}`);
    };

    setLogMethod(log) {
        if (typeof log === 'function'){
            this.trainOpts.log = log;
        } else if (log) {
            this.trainOpts.log = this.logTrainingStatus;
        } else {
            this.trainOpts.log = false;
        }
    }
    validateTrainingOptions(options) {
        const validations = {
            iterations: (val) => { return typeof val === 'number' && val > 0; },
            errorThresh: (val) => { return typeof val === 'number' && val > 0 && val < 1; },
            log: (val) => { return typeof val === 'function' || typeof val === 'boolean'; },
            logPeriod: (val) => { return typeof val === 'number' && val > 0; },
            learningRate: (val) => { return typeof val === 'number' && val > 0 && val < 1; },
            momentum: (val) => { return typeof val === 'number' && val > 0 && val < 1; },
            callback: (val) => { return typeof val === 'function' || val === null },
            callbackPeriod: (val) => { return typeof val === 'number' && val > 0; },
            timeout: (val) => { return typeof val === 'number' && val > 0 }
        };
        for (const p in validations) {
            if (!validations.hasOwnProperty(p)) continue;
            if (!options.hasOwnProperty(p)) continue;
            if (!validations[p](options[p])) {
                throw new Error(`[${p}, ${options[p]}] is out of normal training range, your network will probably not train.`);
            }
        }
    }

    updateTrainingOptions(options) {
        const trainDefaults = this.constructor.trainDefaults;
        for (const p in trainDefaults) {
            if (!trainDefaults.hasOwnProperty(p)) continue;
            this.trainOpts[p] = options.hasOwnProperty(p)
                ? options[p]
                : trainDefaults[p];
        }
        this.validateTrainingOptions(this.trainOpts);
        this.setLogMethod(options.log || this.trainOpts.log);
        this.activation = options.activation || this.activation;
    }
    
    _adjustWeightsAdam() {
        this.iterations++;
    
        const { iterations } = this;
        const {
          beta1,
          beta2,
          epsilon,
          learningRate
        } = this.trainOpts;
    
        for (let layer = 1; layer <= this.outputLayer; layer++) {
            const incoming = this.outputs[layer - 1];
            const currentSize = this.sizes[layer];
            const currentDeltas = this.deltas[layer];
            const currentChangesLow = this.changesLow[layer];
            const currentChangesHigh = this.changesHigh[layer];
            const currentWeights = this.weights[layer];
            const currentBiases = this.biases[layer];
            const currentBiasChangesLow = this.biasChangesLow[layer];
            const currentBiasChangesHigh = this.biasChangesHigh[layer];
    
            for (let node = 0; node < currentSize; node++) {
                const delta = currentDeltas[node];
    
                for (let k = 0; k < incoming.length; k++) {
                    const gradient = delta * incoming[k];
                    const changeLow = currentChangesLow[node][k] * beta1 + (1 - beta1) * gradient;
                    const changeHigh = currentChangesHigh[node][k] * beta2 + (1 - beta2) * gradient * gradient;
            
                    const momentumCorrection = changeLow / (1 - Math.pow(beta1, iterations));
                    const gradientCorrection = changeHigh / (1 - Math.pow(beta2, iterations));
            
                    currentChangesLow[node][k] = changeLow;
                    currentChangesHigh[node][k] = changeHigh;
                    currentWeights[node][k] += learningRate * momentumCorrection / (Math.sqrt(gradientCorrection) + epsilon);
                }
    
                const biasGradient = currentDeltas[node];
                const biasChangeLow = currentBiasChangesLow[node] * beta1 + (1 - beta1) * biasGradient;
                const biasChangeHigh = currentBiasChangesHigh[node] * beta2 + (1 - beta2) * biasGradient * biasGradient;
        
                const biasMomentumCorrection = currentBiasChangesLow[node] / (1 - Math.pow(beta1, iterations));
                const biasGradientCorrection = currentBiasChangesHigh[node] / (1 - Math.pow(beta2, iterations));
        
                currentBiasChangesLow[node] = biasChangeLow;
                currentBiasChangesHigh[node] = biasChangeHigh;
                currentBiases[node] += learningRate * biasMomentumCorrection / (Math.sqrt(biasGradientCorrection) + epsilon);
            }
        }
    }

    _setupAdam() {
        this.biasChangesLow = [];
        this.biasChangesHigh = [];
        this.changesLow = [];
        this.changesHigh = [];
        this.iterations = 0;
    
        for (let layer = 0; layer <= this.outputLayer; layer++) {
            let size = this.sizes[layer];
            if (layer > 0) {
                this.biasChangesLow[layer] = zeros(size);
                this.biasChangesHigh[layer] = zeros(size);
                this.changesLow[layer] = new Array(size);
                this.changesHigh[layer] = new Array(size);
    
                for (let node = 0; node < size; node++) {
                    let prevSize = this.sizes[layer - 1];
                    this.changesLow[layer][node] = zeros(prevSize);
                    this.changesHigh[layer][node] = zeros(prevSize);
                }
            }
        }
    
        this.adjustWeights = this._adjustWeightsAdam;
    }

    initialize() {
        if (!this.sizes) throw new Error ('Sizes must be set before initializing');
    
        this.outputLayer = this.sizes.length - 1;
        this.biases = []; // weights for bias nodes
        this.weights = [];
        this.outputs = [];
    
        // state for training
        this.deltas = [];
        this.changes = []; // for momentum
        this.errors = [];
    
        for (let layer = 0; layer <= this.outputLayer; layer++) {
          let size = this.sizes[layer];
          this.deltas[layer] = zeros(size);
          this.errors[layer] = zeros(size);
          this.outputs[layer] = zeros(size);
    
          if (layer > 0) {
            this.biases[layer] = randos(size);
            this.weights[layer] = new Array(size);
            this.changes[layer] = new Array(size);
    
            for (let node = 0; node < size; node++) {
              let prevSize = this.sizes[layer - 1];
              this.weights[layer][node] = randos(prevSize);
              this.changes[layer][node] = zeros(prevSize);
            }
          }
        }
    
        this.setActivation();
        if (this.trainOpts.praxis === 'adam') {
            this._setupAdam();
        }
    }

    fromJSON(json) {
        Object.assign(this, this.constructor.defaults, json);
        this.sizes = json.sizes;
        this.initialize();
    
        for (let i = 0; i <= this.outputLayer; i++) {
                let layer = json.layers[i];
            if (i === 0 && (!layer[0] || json.inputLookup)) {
                this.inputLookup = lookup.toHash(layer);
                this.inputLookupLength = Object.keys(this.inputLookup).length;
            }
            else if (i === this.outputLayer && (!layer[0] || json.outputLookup)) {
                this.outputLookup = lookup.toHash(layer);
            }
            if (i > 0) {
                const nodes = Object.keys(layer);
                this.sizes[i] = nodes.length;
                for (let j in nodes) {
                    if (nodes.hasOwnProperty(j)) {
                        const node = nodes[j];
                        this.biases[i][j] = layer[node].bias;
                        this.weights[i][j] = toArray(layer[node].weights);
                    }
                }
            }
        }
        if (json.hasOwnProperty('trainOpts')) {
            this.updateTrainingOptions(json.trainOpts);
        }
        return this;
    }

    get isRunnable(){
        if(!this.runInput){
            console.error('Activation function has not been initialized, did you run train()?');
            return false;
        }
    
        const checkFns = [
          'sizes',
          'outputLayer',
          'biases',
          'weights',
          'outputs',
          'deltas',
          'changes',
          'errors',
        ].filter(c => this[c] === null);
    
        if(checkFns.length > 0){
            console.error(`Some settings have not been initialized correctly, did you run train()? Found issues with: ${checkFns.join(', ')}`);
            return false;
        }
        return true;
    }
    
    run(input) {
        if (!this.isRunnable) return null;
        if (this.inputLookup) {
            input = lookup.toArray(this.inputLookup, input, this.inputLookupLength);
        }
    
        let output = this.runInput(input).slice(0);
    
        if (this.outputLookup) {
            output = lookup.toObject(this.outputLookup, output);
        }
        return output;
    }
}
