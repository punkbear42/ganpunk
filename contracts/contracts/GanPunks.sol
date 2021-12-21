// SPDX-License-Identifier: MIT
pragma solidity ^0.8.2;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract GanPunks is ERC721, Ownable {
    using Counters for Counters.Counter;

    Counters.Counter private _tokenIdCounter;

    /*
        configuration
    */
    uint minMintPeriod;
    uint minExtendedMint;
    
    /*
        state
    */
    mapping (bytes32 => address) hashedLatentSpace;
    mapping (uint256 => int[100]) latentSpaces;
    mapping (address => uint) mintPeriod;
    mapping (address => uint) nbMintedToken;
     
    constructor() ERC721("GanPunks", "GANPUNK") {
        minMintPeriod = 10;
        minExtendedMint = 5;
    }
    
    function safeMint(int[100][3] calldata _inputs) public {
        require(block.number - mintPeriod[msg.sender] > minMintPeriod, "the minimum period between minting hasn't been reached"); // minting is only allowed every minMintPeriod blocks.
        mint(_inputs[0], msg.sender);
        if (nbMintedToken[msg.sender] > minExtendedMint) { // extra minting is allowed after minExtendedMint mints.
            mint(_inputs[1], address(this));
            mint(_inputs[2], address(this));
        }
        mintPeriod[msg.sender] = block.number;
    }

    function mint(int[100] calldata _input, address _to) private {
        bytes32 hashedInput = keccak256(abi.encodePacked(_input));
        if (hashedLatentSpace[hashedInput] != address(0)) return;

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        _safeMint(_to, tokenId);
        latentSpaces[tokenId] = _input;
        nbMintedToken[_to]++;
        hashedLatentSpace[hashedInput] = _to;
    }

    function latentSpaceOf(uint tokenId) public view returns (int[100] memory) {
        require(_exists(tokenId), "token does not exist");
        return latentSpaces[tokenId];
    }

    function latentSpaceOwner(int[100] calldata _input) public view returns (address) {
        return hashedLatentSpace[keccak256(abi.encodePacked(_input))];
    }
}