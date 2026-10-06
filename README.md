### Commands

| Command                                        | Mode      | Model                                        |
|------------------------------------------------|-----------|----------------------------------------------|
| ```cargo run```                                      | Training  | New randomly initialized model               |
| ```cargo run -- --model=path/to/model.bin```         | Inference | Loads the saved model                        |
| ```cargo run -- --model=path/to/model.bin --train``` | Training  | Loads the saved model and continues training |
| ```cargo run -- --model=path/to/model.bin --train --epsilon=0.5``` | Training | Set the epsilon value when training starts |