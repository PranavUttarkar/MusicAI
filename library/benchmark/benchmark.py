class Benchmark:
    def __init__(self, benchmark_name, model, datasets):
        self.benchmark_name = benchmark_name
        self.model = model
        self.datasets = datasets

    def run(self):
        results = {}
        for dataset in self.datasets:
            predictions = self.model.test(dataset)
            accuracy, precision, recall, f1 = self.model.evaluate(dataset.labels, predictions)
            results[dataset.name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1
            }
        return results

def get_benchmark(benchmark_name, model, datasets):
    return Benchmark(benchmark_name, model, datasets)