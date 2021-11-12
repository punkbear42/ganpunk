for all the arrays of length 100 (latent space has length 100)
update at index 0, 50, 51, 70 (increment)

```
for x in range(100):
		x_input[x * 100] = x_input[x * 100] + x
		x_input[(x * 100) + 50] = x_input[(x * 100) + 50] + x
		x_input[(x * 100) + 51] = x_input[(x * 100) + 51] + x
		x_input[(x * 100) + 70] = x_input[(x * 100) + 51] + x
```