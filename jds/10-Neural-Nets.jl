### A Pluto.jl notebook ###
# v0.20.4

using Markdown
using InteractiveUtils

# ╔═╡ 14ee59c0-c13c-11ef-3d48-6f75264fc65a
begin
	using Flux
	using Flux: onehotbatch, argmax, crossentropy, throttle
	using Base.Iterators: repeated
	using MLDatasets: MNIST
	using Images
	using ImageInTerminal
end

# ╔═╡ 8c92551a-2262-410c-bb26-06da69e48f35
md"Main package is `Flux.jl`"

# ╔═╡ 3411f7ed-dddb-4b99-875c-74d9d22f4c91
md"We'll load the `MNIST` dataset using the `MLDatasets` package"

# ╔═╡ e2ab7705-e386-4828-b202-f6b3781ccaec
fieldnames(MNIST)

# ╔═╡ 38608169-fa97-4577-94c8-ae5b0e3f1645
imgs = MNIST(Float32, :train, dir="data/mnist")

# ╔═╡ 145a3a2d-2f1a-4441-babb-d399e5a4eafd
imgs.features


# ╔═╡ 2ce227ea-ea0b-431b-9f8c-cc936403366a
typeof(imgs[3])

# ╔═╡ 24b1d599-fd7e-40ea-96b0-2171605fd989
Gray.(imgs.features[:,:,14])

# ╔═╡ 5bf25035-1ebd-4a6e-b9c7-6a4cd3fd081c
colorview(Gray, imgs.features[:,:,3])

# ╔═╡ ec74003f-f966-475e-bd0b-61f5ed0224f1
trainfig = vec(imgs.features[:,:,3])

# ╔═╡ 5e582af8-8b31-455d-b2ca-c1d84adecd0f
yy = onehotbatch(imgs.targets, 0:9)

# ╔═╡ 1401c08c-1059-47cb-9c18-2590d60634c2
model = Chain(
	Dense(28^2, 32, relu),
	Dense(32, 10), softmax)

# ╔═╡ 7b06ceb4-2c48-4fb5-a98f-2b22c6ec6b16
# Loss function and accuracy
begin
	loss(m, x, y) = Flux.crossentropy(m(x), y)
	accuracy(m, x, y) = mean(argmax(m(x)) .== argmax(y))
end

# ╔═╡ 2413aa61-be5f-4fcc-bbaa-7a3195a1c4ad
size(imgs.features)

# ╔═╡ 4c099a9a-b640-412e-8d75-a69b6f4a453d
xx = reshape(imgs.features, :, 60000)

# ╔═╡ 3e35b044-e751-4ef3-993d-039351c05376
begin
    datasetx = repeated((xx, yy), 200)
	cc = collect(datasetx)
end

# ╔═╡ 64aba333-8030-44f8-ab8f-71b85b1bcf97
evalcb = () -> @show(loss(model, xx, yy))

# ╔═╡ dbe88185-c8e0-450e-9cd1-65f8b8e2b44d
ps = Flux.trainables(model)

# ╔═╡ 12880055-679b-4838-b1d2-9b3caa1a8485
ps2 = gradient(m -> loss(m, xx, yy), model)

# ╔═╡ 3b8ec00b-2955-4bd0-a5ca-158463f82fd9
opt_state = Flux.setup(Adam(), model)

# ╔═╡ d97c78f4-30a0-4972-ba65-71b74af57915
Flux.train!(loss, model, datasetx, opt_state,)

# ╔═╡ 01805d64-2e03-4552-8bef-28fd07aa2765
loss(model, xx, yy)

# ╔═╡ ab72b02b-8c4a-4318-90d3-d5ec7f4ae337
imgtest = MNIST(Float32, :test, dir="data/mnist")

# ╔═╡ 452ee9af-df10-40d4-8d15-4c9c2341b03d
txx = reshape(imgtest.features, :, 10000)

# ╔═╡ 7ff871a9-e2dd-4dc5-b6d5-e899857c8806
testimg = model(txx[:,1])

# ╔═╡ 1dd0692a-87c9-4a80-928f-75c38b895f8c
argmax(testimg) - 1

# ╔═╡ a559ba93-8a7c-4c1b-9d04-8173aa18a7a2
colorview(Gray, reshape(txx[:,1], 28, 28))

# ╔═╡ 8156ad72-e3d7-414a-92fd-32ff9df53df9
colorview(Gray, imgtest.features[:,:,1])

# ╔═╡ e6f5c3c6-f5a0-4069-9e30-9fbc36a9f185
model(trainfig)

# ╔═╡ b9e785d4-32f9-4fe6-b929-bb3f9b0fcadc
model(xx[:, 2])

# ╔═╡ 6f48ddfe-2a39-4540-a7f2-d2b0603d7cfc
Int.(yy[:, 2])

# ╔═╡ 40310029-da92-4225-91ba-1a053203ced4
testfig = vec(imgtest.features[:,:,3])

# ╔═╡ 127f8210-10e7-4d43-bf71-26f4e697ab5c
# untrained model
model(testfig)

# ╔═╡ d223b2c5-3203-49bc-9c89-bbfd60188c85
md"""

### Cool stuff

This notebook was done with the more up-to-date API of `Flux` and `MLDatasets` compared to the version of the packages used in the original notebook.

"""

# ╔═╡ b4027277-b4c2-4ede-90c5-3252aed37fa2


# ╔═╡ Cell order:
# ╟─8c92551a-2262-410c-bb26-06da69e48f35
# ╠═14ee59c0-c13c-11ef-3d48-6f75264fc65a
# ╟─3411f7ed-dddb-4b99-875c-74d9d22f4c91
# ╠═e2ab7705-e386-4828-b202-f6b3781ccaec
# ╠═38608169-fa97-4577-94c8-ae5b0e3f1645
# ╠═145a3a2d-2f1a-4441-babb-d399e5a4eafd
# ╠═2ce227ea-ea0b-431b-9f8c-cc936403366a
# ╠═24b1d599-fd7e-40ea-96b0-2171605fd989
# ╠═5bf25035-1ebd-4a6e-b9c7-6a4cd3fd081c
# ╠═ec74003f-f966-475e-bd0b-61f5ed0224f1
# ╠═5e582af8-8b31-455d-b2ca-c1d84adecd0f
# ╠═1401c08c-1059-47cb-9c18-2590d60634c2
# ╠═127f8210-10e7-4d43-bf71-26f4e697ab5c
# ╠═7b06ceb4-2c48-4fb5-a98f-2b22c6ec6b16
# ╠═2413aa61-be5f-4fcc-bbaa-7a3195a1c4ad
# ╠═4c099a9a-b640-412e-8d75-a69b6f4a453d
# ╠═3e35b044-e751-4ef3-993d-039351c05376
# ╠═64aba333-8030-44f8-ab8f-71b85b1bcf97
# ╠═dbe88185-c8e0-450e-9cd1-65f8b8e2b44d
# ╠═12880055-679b-4838-b1d2-9b3caa1a8485
# ╠═3b8ec00b-2955-4bd0-a5ca-158463f82fd9
# ╠═d97c78f4-30a0-4972-ba65-71b74af57915
# ╠═01805d64-2e03-4552-8bef-28fd07aa2765
# ╠═ab72b02b-8c4a-4318-90d3-d5ec7f4ae337
# ╠═452ee9af-df10-40d4-8d15-4c9c2341b03d
# ╠═7ff871a9-e2dd-4dc5-b6d5-e899857c8806
# ╠═1dd0692a-87c9-4a80-928f-75c38b895f8c
# ╠═a559ba93-8a7c-4c1b-9d04-8173aa18a7a2
# ╠═8156ad72-e3d7-414a-92fd-32ff9df53df9
# ╠═e6f5c3c6-f5a0-4069-9e30-9fbc36a9f185
# ╠═b9e785d4-32f9-4fe6-b929-bb3f9b0fcadc
# ╠═6f48ddfe-2a39-4540-a7f2-d2b0603d7cfc
# ╠═40310029-da92-4225-91ba-1a053203ced4
# ╟─d223b2c5-3203-49bc-9c89-bbfd60188c85
# ╠═b4027277-b4c2-4ede-90c5-3252aed37fa2
