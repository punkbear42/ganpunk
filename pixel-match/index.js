const fs = require('fs');
const PNG = require('pngjs').PNG;
const pixelmatch = require('pixelmatch');
 


function diff (ite) {
    const path1 = '../punks/punk_' + ite + '.png'
    const path2 = '../results/generated_plot_e486_000.png'
    const img1 = PNG.sync.read(fs.readFileSync(path1));
    const img2 = PNG.sync.read(fs.readFileSync(path2));
    const {width, height} = img1;
    const diff = new PNG({width, height});
    
    const numDiffPixels = pixelmatch(img1.data, img2.data, diff.data, width, height, {threshold: 0.1});
    console.log(ite, numDiffPixels)
    return {numDiffPixels, ite}
    // fs.writeFileSync('diff.png', PNG.sync.write(diff));
}

let min = {numDiffPixels:10000, ite:-1}
let aggregator = {}
for (let k = 0; k < 10000; k++) {
    const diffPixels = diff(k)
    if  (diffPixels.numDiffPixels < min.numDiffPixels) { min = diffPixels; }
    if (!aggregator[diffPixels.numDiffPixels]) aggregator[diffPixels.numDiffPixels] = { incr: 0, punks: [] }
    aggregator[diffPixels.numDiffPixels].incr = aggregator[diffPixels.numDiffPixels].incr + 1
    aggregator[diffPixels.numDiffPixels].punks.push(k)
}
console.log('min', min)
console.log(aggregator)
