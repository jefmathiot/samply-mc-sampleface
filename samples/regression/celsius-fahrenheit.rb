# frozen_string_literal: true

require "neuronface"
require "gruff"

data = [
  [[-40.0], [-40.0]],
  [[-10.0], [14.0]],
  [[0.0], [32.0]],
  [[8.0], [46.4]],
  [[15.0], [59.0]],
  [[22.0], [71.6]],
  [[38.0], [100.4]]
]

dataset = Neuronface::Datasets.get(:memory)
                              .new(data)
                              .normalize!
                              .shuffle!

model = Neuronface::Model.new(loss: :squared_error)
                         .append(:input, 1)
                         .append(:dense, 1, activation: :relu)

history = model.fit(dataset, :simple, epochs: 20)

puts dataset.outputs_normalizer.revert(model.predict(dataset.inputs_normalizer.convert([100])))

def plot_labels(history)
  label_count = [10, history[:loss].size].min
  label_count.times
             .map { |i| i * history[:loss].size / label_count }
             .each_with_object({}) { |value, hash| hash[value] = value.to_s }
end

def plot(history)
  g = Gruff::Line.new("1200x600")
  g.labels = plot_labels(history)
  g.title = "Model training"
  g.data "Training loss", history[:loss]
  Dir.mkdir "tmp"
  g.write "tmp/celsius-fahrenheit.png"
end

plot(history)
