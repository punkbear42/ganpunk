// SPDX-License-Identifier: MIT
pragma solidity ^0.8.2;

import "@openzeppelin/contracts/token/ERC721/ERC721.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721URIStorage.sol";
import "@openzeppelin/contracts/token/ERC721/extensions/ERC721Enumerable.sol";
import "@openzeppelin/contracts/token/ERC721/IERC721Receiver.sol";
import "@openzeppelin/contracts/access/Ownable.sol";
import "@openzeppelin/contracts/utils/Counters.sol";

contract GanPunks is ERC721, ERC721Enumerable, Ownable, IERC721Receiver {
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
    mapping (uint256 => string[100]) latentSpaces;
    mapping (address => uint) mintPeriod;
    mapping (address => uint) nbMintedToken;
     
    constructor() ERC721("GanPunks", "GANPUNK") {
        minMintPeriod = 10;
        minExtendedMint = 5;
    }
    
    /*
        minting is allowed only every "minMintPeriod" block.
        first nft is minted for the spender
        the two others are minted to the contract, which will immediately free them to an auction.
    */
    function safeMint(string[100][3] calldata _inputs) public {
        require(block.number - mintPeriod[msg.sender] > minMintPeriod, "the minimum period between minting hasn't been reached"); // minting is only allowed every minMintPeriod blocks.
        mint(_inputs[0], msg.sender);
        if (nbMintedToken[msg.sender] > minExtendedMint) { // extra minting is allowed after minExtendedMint mints.
            mint(_inputs[1], address(this));
            mint(_inputs[2], address(this));
        }
        mintPeriod[msg.sender] = block.number;
    }

    function mint(string[100] memory _input, address _to) private {
        bytes32 hashedInput = keccak256(abi.encode(_input));
        if (hashedLatentSpace[hashedInput] != address(0)) return;

        uint256 tokenId = _tokenIdCounter.current();
        _tokenIdCounter.increment();
        _safeMint(_to, tokenId);
        latentSpaces[tokenId] = _input;
        nbMintedToken[_to]++;
        hashedLatentSpace[hashedInput] = _to;
    }

    function latentSpaceOf(uint tokenId) public view returns (string[100] memory) {
        require(_exists(tokenId), "token does not exist");
        return latentSpaces[tokenId];
    }

    function latentSpaceOwner(int[100] calldata _input) public view returns (address) {
        return hashedLatentSpace[keccak256(abi.encodePacked(_input))];
    }

    // The following functions are overrides required by Solidity.

    function _beforeTokenTransfer(address from, address to, uint256 tokenId)
        internal
        override(ERC721, ERC721Enumerable)
    {
        super._beforeTokenTransfer(from, to, tokenId);
    }

    function supportsInterface(bytes4 interfaceId)
        public
        view
        override(ERC721, ERC721Enumerable)
        returns (bool)
    {
        return super.supportsInterface(interfaceId);
    }

    function onERC721Received(address operator, address from, uint256 tokenId, bytes memory data)
        public
        override(IERC721Receiver)
        returns (bytes4)
    {
        return this.onERC721Received.selector;
    }
}
