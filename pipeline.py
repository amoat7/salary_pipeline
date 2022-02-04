from cmath import pi
from tfx.proto import example_gen_pb2
from tfx.orchestration import pipeline 
import os  
from tfx.components import CsvExampleGen 
from tfx.components import StatisticsGen
from tfx.components import SchemaGen 
from tfx.components import ExampleValidator
from tfx.components import Transform 



census_transform_module_file = 'census_transform.py'

def create_pipeline(
    pipeline_name,
    pipeline_root,
    data_path,
    serving_dir,
    beam_pipeline_args = None,
    metadata_connection_config=None
):
    components =[]

    #example gen
    output = example_gen_pb2.Output(
        split_config=example_gen_pb2.SplitConfig(splits=[
            example_gen_pb2.SplitConfig.Split(name='train', hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name='eval', hash_buckets=2)
        ])
    )

    example_gen = CsvExampleGen(input_base=data_path, output_config=output)
    components.append(example_gen)

    statistics_gen = StatisticsGen(examples=example_gen.outputs['examples'])
    components.append(statistics_gen)

    schema_gen = SchemaGen(statistics=statistics_gen.outputs['statistics'])
    components.append(schema_gen)

    validator  = ExampleValidator(
        statistics=statistics_gen.outputs['statistics'],
        schema=schema_gen.outputs['schema']
    )
    components.append(validator)

    transform  = Transform(
        examples=example_gen.outputs['examples'],
        schema=schema_gen.outputs['schema'],
        module_file=census_transform_module_file
    )
    components.append(transform)

    



    return pipeline.Pipeline(

        pipeline_name=pipeline_name,
        pipeline_root=pipeline_root,
        components=components,
        beam_pipeline_args=beam_pipeline_args,
        metadata_connection_config=metadata_connection_config
    )
