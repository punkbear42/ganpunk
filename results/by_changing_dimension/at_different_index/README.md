for all the arrays of length 100 (latent space has length) 100
update at different index at each iteration

```
for x in range(100):
		x_input[x * 100] = x_input[x * 100] + x
		x_input[x * 98] = x_input[x * 98] + x
		x_input[x * 99] = x_input[x * 99] + x		
```