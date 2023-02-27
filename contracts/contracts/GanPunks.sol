// SPDX-License-Identifier: MIT
pragma solidity ^0.8.4;

import "@uniswap/v3-periphery/contracts/interfaces/ISwapRouter.sol";
import "@uniswap/v3-periphery/contracts/interfaces/IQuoter.sol";
import "@openzeppelin/contracts-upgradeable/token/ERC721/ERC721Upgradeable.sol";
import "@openzeppelin/contracts-upgradeable/access/OwnableUpgradeable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/Initializable.sol";
import "@openzeppelin/contracts-upgradeable/proxy/utils/UUPSUpgradeable.sol";

import "hardhat/console.sol";

contract GanPunk is Initializable, ERC721Upgradeable, OwnableUpgradeable, UUPSUpgradeable {    
    // ISwapRouter public constant uniswapRouter = ISwapRouter(0xE592427A0AEce92De3Edee1F18E0157C05861564);
    IQuoter public constant quoter = IQuoter(0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6);
    address private constant DAI = 0x6B175474E89094C44Da98b954EedeAC495271d0F;
    address private constant WETH9 = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;

    mapping (bytes32 => address) public hashedLatentSpace;
    mapping (uint256 => string[100]) public latentSpaces;

    uint creationCostInDai;

    address payable safe;

    /// @custom:oz-upgrades-unsafe-allow constructor
    constructor() {
        _disableInitializers();
    }

    function initialize() initializer public {
        __ERC721_init("GanPunk", "GPunk");
        __Ownable_init();
        __UUPSUpgradeable_init();
        creationCostInDai = 50;
    }

    function _authorizeUpgrade(address newImplementation)
        internal
        onlyOwner
        override
    {}

    function setSafe(address payable _safe) public onlyOwner {
        safe = _safe;
    }

    function mint(string[100] memory _input, address _to, uint256 tokenId) public payable {
        require(0 != msg.value, "value is set to 0");
        require(safe != address(0), "safe not set");
        require(safe.send(msg.value), "failed forwarding payment");
        bytes32 hashedInput = keccak256(abi.encode(_input));
        require(hashedLatentSpace[hashedInput] == address(0), "input already assigned");

        _safeMint(_to, tokenId);
        latentSpaces[tokenId] = _input;
        hashedLatentSpace[hashedInput] = _to;        
    }

    function latentSpaceOf(uint tokenId) public view returns (string[100] memory) {
        require(_exists(tokenId), "token does not exist");
        return latentSpaces[tokenId];
    }

    function latentSpaceOwner(string[100] calldata _input) public view returns (address) {
        return hashedLatentSpace[keccak256(abi.encode(_input))];
    }

    function getEstimatedETHforDAI(uint daiAmount) public returns (uint256) {
        address tokenIn = DAI;
        address tokenOut = WETH9;
        uint24 fee = 500;
        uint160 sqrtPriceLimitX96 = 0;

        return quoter.quoteExactInputSingle(
            tokenIn,
            tokenOut,
            fee,
            daiAmount,
            sqrtPriceLimitX96
        );
    }
}
