/**
 * Training is done on pre-trained mobilenet model for recoginizing images
 * Flow of data
 * images -> (mobilenet model) -> output -> (new_customized_model) -> classification
 */

let mobilenet;
let model;
const rps_data = new RPSData();                                     // RPS data class
const webcam = new Webcam(document.getElementById('wc'));           // webcam class in Webcam.js
var rockSamples=0, paperSamples=0, scissorsSamples=0;
var startPredictionsInterval;
var time = 1000;
var computerScore=0, playerScore=0;

async function init(){
    await webcam.setup();
    mobilenet = await loadMobilenet();
    // loading weights takes time mean while we capture the images and throw away results to avoid lag
    tf.tidy(() => mobilenet.predict(webcam.capture()));

}

// loading pre-trained model
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');                          // get output of 'conv_pw_13_relu' layer
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

// new model - inputs are output of the pre-trained model
async function train(){
    rps_data.ys = null
    rps_data.convertLabels(3);

    model = tf.sequential({
        layers:[
            tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
            tf.layers.dense({ units: 100, activation: 'relu'}),
            tf.layers.dense({ units: 3, activation: 'softmax'})                 // classfies into three classes
        ]
    });

    model.compile({optimizer: tf.train.adam(0.0001), loss: 'categoricalCrossentropy', metrics: ['accuracy']});
    let loss = 0;
    model.fit(rps_data.xs, rps_data.ys, {
        epochs: 10,
        callbacks: {
            onBatchEnd: async (batch, logs) => {
                console.log('loss ' + logs.loss.toFixed(5) + ' accuracy ' + logs.acc);
            }
        }
    })
}

// getting data
function handleButton(element){
	switch(element.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
    const image = webcam.capture();
	label = parseInt(element.id);
	rps_data.addImage(mobilenet.predict(image), label);         // passing image into mobilenet model and getting output weights #transfer learning
}

// making predictions
async function predicts(makePrediction) {
    if(makePrediction){
        startPredictionsInterval = setInterval(function() {
            doPredictions();
        }, time);
    } else clearInterval(startPredictionsInterval);

}


var previousClass = -1;
async function doPredictions(){
    const predictClassRPS = tf.tidy(() => {
        const image = webcam.capture();
        const mobilenetOutput = mobilenet.predict(image);
        const prediction = model.predict(mobilenetOutput);
        // console.log(prediction);
        return prediction.as1D().argMax();      // return 1D tensor containing prediction
    });

    // players Move
    const predictedClassID = (await predictClassRPS.data())[0];
    var text = "";
    switch (predictedClassID) {
        case 0:
            text = "PL Rock";
            break;
        case 1:
            text = "PL Paper";
            break;
        case 2:
            text = "PL Scissors";
            break;
    }

    // change only when user changes move
    if (predictedClassID != previousClass){
        document.getElementById("prediction").innerText = text;

        // computers Move
        var computerText = "";
        var computersMove = Math.floor((Math.random() * 3));
        if (computersMove == 0) computerText = "CP Rock";
        if (computersMove == 1) computerText = "CP Paper";
        if (computersMove == 2) computerText = "CP Scissors";

        setComputersMove(computerText);

        // get winner
        var result = getWinner(predictedClassID, computersMove);
        document.getElementById("winner").innerText = "WINNER " + result;

        // score
        if(result.localeCompare("PLAYER")) computerScore++;
        if(result.localeCompare("COMPUTER")) playerScore++;
        document.getElementById("score").innerText = "Computer's Score " + computerScore + " / " + "Player's Score " + playerScore ;

    }
    previousClass = predictedClassID;

    // dispose predicted class
    predictClassRPS.dispose();
    await tf.nextFrame();
}


function getWinner(playersMove, computersMove){
    console.log(playersMove, computersMove);

    if(playersMove == computersMove) return "TIE";
    if(playersMove == 0 && computersMove == 2) return "PLAYER";
    if(playersMove == 1 && computersMove == 0) return "PLAYER";
    if(playersMove == 2 && computersMove == 1) return "PLAYER";
    if(playersMove == 0 && computersMove == 1) return "COMPUTER";
    if(playersMove == 1 && computersMove == 2) return "COMPUTER" ;
    if(playersMove == 2 && computersMove == 0) return "COMPUTER";

}

function doTraining(){
    if (rockSamples > 0 | paperSamples > 0 | scissorsSamples > 0){
        train();
        alert('Training Completed');
    } else alert('Add Training Dataset');
}

function startPredicting(){
    if (rockSamples > 0 | paperSamples > 0 | scissorsSamples > 0){
        predicts(true);
    } else alert('Add Training Dataset');
}

function stopPredicting(){
    if (rockSamples > 0 | paperSamples > 0 | scissorsSamples > 0){
        predicts(false);
    } else alert('Add Training Dataset');
}

function resetScore(){
    predicts(false);
    computerScore=0;
    playerScore=0;
    document.getElementById("prediction").innerText = "";
    document.getElementById("score").innerText = "";
    document.getElementById("winner").innerText = "";
    setComputersMove("");
}

function resetAll(){
    rockSamples=0;
    paperSamples=0;
    scissorsSamples=0;
    document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
    document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
    document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
    resetScore();
    init();
}

function setComputersMove(msg){
    document.getElementById("computer").innerText = msg;
}

init();
