const { assert, expect } = require("chai");
const { ethers } = require("ethers");
const BigNumber = require('bignumber.js');
import { deployProxy } from './ethers-lib'

let gan
describe("GanPunk", function () { 
  describe("Deployment", function () {
    it("should deploy", async () => {
     const [owner] = await ethers.getSigners();
      gan = await deployProxy('GanPunk', 'ERC1967Proxy', owner, [])
      await gan.deployed();
      console.log("gan deployed to:", gan.address);      
    })
  })

  describe("Basic Checks", function () {
    it("should call getEstimatedETHforDAI", async () => {
      const cost = new BigNumber(50)
      const estimatedEth = await gan.callStatic.getEstimatedETHforDAI(cost.multipliedBy(1000000).toString());
      let weth = new BigNumber(estimatedEth.toString()).multipliedBy(new BigNumber(1000000000000000000)).dividedBy(new BigNumber(1000000))
      weth = weth.toNumber()
      console.log("estimated weth", weth);
      console.log("estimated eth", weth / 1000000000000000000)
      assert.isTrue(weth / 1000000000000000000 > 0)
    })

    it("should set the safe", async () => {
      const [owner] = await ethers.getSigners();
      const setSafe = await gan.setSafe(owner.address)
      setSafe.wait()
    })
  })

  const inputs = []
  for (let i = 0; i < 100; i++) { inputs.push(Math.random().toString()) }
  describe("Minting", function () {

    it("should mint", async () => {
      const [owner] = await ethers.getSigners();
      const mint = await expect(gan.mint(inputs, owner.address, 0, { value: 0 })).to.be.revertedWith('value is set to 0');      
    })

    it("should mint", async () => {
      const [owner] = await ethers.getSigners();
      const mint = await gan.mint(inputs, owner.address, 0, { value: 1 })
      mint.wait()
      const ownerFirst = await gan.ownerOf(0)
      assert.equal(ownerFirst, owner.address)      
    })

    it("should retrieve the latent space", async () => {
      const inputsCheck = await gan.latentSpaceOf(0)
      assert.deepEqual(inputsCheck, inputs)
    })

    it("should retrieve the owner", async () => {
      const [owner] = await ethers.getSigners();
      let ownerCheck = await gan.latentSpaceOwner(inputs)
      assert.deepEqual(ownerCheck, owner.address)

      const inputsNoOwner = []
      for (let i = 0; i < 100; i++) { inputsNoOwner.push(Math.random().toString()) }
      ownerCheck = await gan.latentSpaceOwner(inputsNoOwner)
      assert.deepEqual(ownerCheck, '0x0000000000000000000000000000000000000000')
    })

    it("should check for a non used input", async () => {
      const inputsNoOwner = []
      for (let i = 0; i < 100; i++) { inputsNoOwner.push(Math.random().toString()) }
      ownerCheck = await gan.latentSpaceOwner(inputsNoOwner)
      assert.deepEqual(ownerCheck, '0x0000000000000000000000000000000000000000')
    })

    it("should throw while minting", async () => {
      const [owner] = await ethers.getSigners();
      await expect(gan.mint(inputs, owner.address, 0, { value: 1 })).to.be.revertedWith('input already assigned');
    })

    it("should mint another one", async () => {
      const inputsSecond = []
      for (let i = 0; i < 100; i++) { inputsSecond.push(Math.random().toString()) }

      const [owner] = await ethers.getSigners();
      const mint = await gan.mint(inputsSecond, owner.address, 1, { value: 1 })
      mint.wait()
      const ownerFirst = await gan.ownerOf(1)
      assert.equal(ownerFirst, owner.address)    
      
      const inputsCheck = await gan.latentSpaceOf(1)
      assert.deepEqual(inputsCheck, inputsSecond)
    })

    it("should throw an error whil minting on a already existing token", async () => {
      const inputsSecond = []
      for (let i = 0; i < 100; i++) { inputsSecond.push(Math.random().toString()) }

      const [owner] = await ethers.getSigners();
      await expect(gan.mint(inputsSecond, owner.address, 0, { value: 1 })).to.be.revertedWith('ERC721: token already minted');
    })
  })
});
