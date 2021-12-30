// Right click on the script name and hit "Run" to execute

const randomFloat = function () {
    const int = window.crypto.getRandomValues(new Uint32Array(1))[0]
    return int / 2**32
  }
  function getRandomIntInclusive(min, max) {
      min = Math.ceil(min);
      max = Math.floor(max);
      return randomFloat() * (max - min + 1) + min //The maximum is inclusive and the minimum is inclusive
  }
  
  function latentSpace () {
      let input = []
      for (let k = 0 ; k < 100; k++) {
          input.push((getRandomIntInclusive(-4, 4) * 10000000000000000000).toString())
      }
      console.log(input);
      return input
  }
  
  (async () => {
      try {
          console.log('minting...')
      
          const contractName = 'GanPunks' // Change this for other contract
          const address = '0xd9145CCE52D386f254917e481eB44e9943F39138';
  
          // Note that the script needs the ABI which is generated from the compilation artifact.
          // Make sure contract is compiled and artifacts are generated
          const artifactsPath = `browser/contracts/artifacts/${contractName}.json` // Change this for different path
      
          const metadata = JSON.parse(await remix.call('fileManager', 'getFile', artifactsPath))
          // 'web3Provider' is a remix global variable object
          const signer = (new ethers.providers.Web3Provider(web3Provider)).getSigner()
      
          let contract = new ethers.Contract(address, metadata.abi, signer);
  
          const tx = await contract.safeMint([latentSpace(), latentSpace(), latentSpace()]);
          console.log('hash', tx.hash);        
      } catch (e) {
          console.log('error', e.message)
      }
  })()