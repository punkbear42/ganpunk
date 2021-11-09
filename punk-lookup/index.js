import csv from 'csv-parser'
import fs from 'fs'

const results = {}

const action = (data, name) => {
    if (!results[name]) results[name] = []
    results[name].push(data.id)
    try {
        fs.mkdirSync('/home/punkbear/punk/punks-classified/' + name.trim().toLowerCase())    
    } catch (e) {}
    fs.copyFile(
        '/home/punkbear/punk/punks/punk_' + data.id + '.png',
        '/home/punkbear/punk/punks-classified/' + name.trim().toLowerCase() + '/punk_' + data.id + '.png',
        (err) => {
            if (err) throw err;
        });    
}

const process = (data, item) => {
    const name = data[item]
    if (!name.trim()) return
    if (name.indexOf('/') !== -1) {
        let splitted = name.split('/')
        for (const k of splitted) {
            action(data, k)
        }
    } else {
        action(data, name)
    }
    
}

const parse = (file) => {
    return new Promise((resolve, reject) => {
        fs.createReadStream(file)
        .pipe(csv())
        .on('data', (data) => {
            (function (data) {
                process(data, ' type')
                process(data, ' gender')
                process(data, ' skin tone')
                process(data, ' accessories')
            })(data)
            
        })
        .on('end', () => {
            resolve()
        });
    })
}

(async function () {
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/0-999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/1000-1999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/2000-2999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/3000-3999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/4000-4999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/5000-5999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/6000-6999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/7000-7999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/8000-8999.csv')
    await parse('/home/punkbear/punk/punk-attributes/punks.attributes/original/9000-9999.csv')
    console.log(results)
    fs.writeFile('punks.json', JSON.stringify(results), () => {})
})()


