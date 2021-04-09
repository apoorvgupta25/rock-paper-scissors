class RPSData {
    constructor() {
        this.labels = []                        // initialize class with empty Labels first
    }

    // adding output prediction from mobilenet model and its label
    addImage(img, label){
        if(this.xs == null){                    // for first sample
            this.xs = tf.keep(img);             // keep the prediction otherwise throw away by tf.tidy()
            this.labels.push(label);
        } else{
            const oldX = this.xs;
            this.xs = tf.keep(oldX.concat(img, 0));
            this.labels.push(label);
            oldX.dispose();
        }
    }

    // creating the targer class (one hot encoded 1,0,0)
    convertLabels(numClasses) {
        for (var i = 0; i < this.labels.length; i++) {
            if (this.ys == null) {
                this.ys = tf.keep(tf.tidy(() => { return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses) }));
            }
            else {
                const y = tf.tidy(() => { return tf.oneHot(tf.tensor1d([this.labels[i]]).toInt(), numClasses) });
                const oldY = this.ys;
                this.ys = tf.keep(oldY.concat(y, 0));
                oldY.dispose();
                y.dispose();
            }
        }
    }

}
