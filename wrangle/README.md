# WrangleConf

On July 28, 2016, I gave a lightning talk at WrangleConf about Fizz Buzz in
Tensorflow.

[slides](https://docs.google.com/presentation/d/1d9uxeT1Kl7kX2xbGj6EhnbaN0ygOl4XcZQlTqYX1k2g/edit?usp=sharing)

How did I turn a joke blog post into a talk? Very carefully. I ran
a bunch of variations and discussed the results.

If you want to run it yourself, here's the crappy workflow I used:

```
In [4]: run 05-deep-learning.py --num_hidden 1000 --num_hidden2 100 --num_epochs 500
0 0.132177681473 1.38595
[[  0.   0.   0.   0.]
 [  0.   0.   0.   0.]
 [ 53.  27.  14.   6.]
 [  0.   0.   0.   0.]]
['buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz'
 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz' 'buzz']
100 0.534127843987 1.04009
200 0.783315276273 0.657773
300 0.962080173348 0.431493
400 0.975081256771 0.361589
[[ 49.   1.   0.   0.]
 [  4.  26.   0.   0.]
 [  0.   0.  14.   0.]
 [  0.   0.   0.   6.]]
['1' '2' 'fizz' '4' 'buzz' 'fizz' '7' '8' 'fizz' 'buzz' '11' 'fizz' '13'
 '14' 'fizzbuzz' '16' '17' 'fizz' 'fizz' 'buzz' 'fizz' '22' '23' 'fizz'
 'buzz' '26' 'fizz' '28' '29' 'fizzbuzz' '31' '32' 'fizz' 'fizz' 'buzz'
 'fizz' '37' 'fizz' 'fizz' 'buzz' '41' 'fizz' '43' '44' 'fizzbuzz' '46'
 '47' 'fizz' '49' 'buzz' 'fizz' '52' '53' 'fizz' 'buzz' '56' 'fizz' '58'
 '59' 'fizzbuzz' '61' '62' 'fizz' '64' 'buzz' 'fizz' '67' 'fizz' 'fizz'
 'buzz' '71' 'fizz' '73' '74' 'fizzbuzz' '76' '77' 'fizz' '79' 'buzz'
 'fizz' '82' '83' 'fizz' 'buzz' '86' 'fizz' '88' '89' 'fizzbuzz' '91' '92'
 'fizz' '94' 'buzz' '96' '97' '98' 'fizz' 'buzz']

In [5]: w_h1, w_h2, w_o, b_h1, b_h2, b_o = learned_parameters

In [6]: b_o
Out[6]: array([-0.06437607, -0.08173542,  0.076047  ,  0.24295421], dtype=float32)
```

Every 100 epochs it prints out the accuracy and error on the training set.
Every 1000 epochs (and when it's done) it prints out its predictions on the test
set and a crosstab of the results.

It also sticks its learned parameter values in `learned_parameters` so you can
play around with them. It's not my best software engineering work, so don't judge me.
