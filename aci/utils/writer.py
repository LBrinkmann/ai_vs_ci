from torch.utils.tensorboard import SummaryWriter
import collections
import scipy.misc
import jsonlines
import torch as th
import datetime
import pandas as pd
import numpy as np
import imageio
import os
import json
from moviepy.editor import ImageSequenceClip

# hack to suppress annoying warnings
import imageio.core.util
def silence_imageio_warning(*args, **kwargs):
    pass
imageio.core.util._precision_warn = silence_imageio_warning


def parse_value(value):
    if type(value) is th.Tensor:
        if value.dim() == 0:
            return value.item()
        elif len(value) == 1:
            return value.item()
        else:
            return [i.item() for i in value]
    elif type(value) is datetime.datetime:
        return value.isoformat()
    else:
        return value

def parse_dict(event_dict):
    return {k: parse_value(v) for k,v in event_dict.items()}

def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def selector(func):
    def wrapper(self, *args, on=None, **kwargs):
        if on is None or ((on in self.periods) and (self.step % self.periods[on]) == 0):
            return func(self, *args, **kwargs)
        else:
            return None
    return wrapper

class Writer():
    def __init__(self, output_folder, periods, flush_period, use_tb=False, **meta):
        self.meta = meta
        ensure_dir(output_folder)
        self.metric_rows = {}
        self.meta_rows = {}
        self.metrics_folder = f"{output_folder}/metrics"
        # self.json_writer = jsonlines.open(f"{output_folder}/metrics.jsonl", mode='w', flush=True)
        if use_tb:
            self.tensorboard_writer = SummaryWriter(log_dir=f'{output_folder}/tensorboard')
        else:
            self.tensorboard_writer = None
        self.image_folder = f"{output_folder}/images"
        self.env_folder = f"{output_folder}/envs"
        self.video_folder = f"{output_folder}/videos"
        self.model_folder = f"{output_folder}/models"
        self.df_folder = f"{output_folder}/df"
        ensure_dir(self.df_folder)
        ensure_dir(self.metrics_folder)
        ensure_dir(self.env_folder)
        self.periods = periods
        self.frames = {}
        self.traces = {}
        self.flush_idx = 0
        self.flush_period = flush_period

    def add_meta(self, **meta):
        self.meta = {**self.meta, **meta}

    @selector
    def check_on(self):
        return True

    @property
    def step(self):
        return self.meta['_step']

    @selector
    def add_table(self, **kwargs):
        # might change in the future

        # now = datetime.datetime.utcnow()
        # meta = {**self.meta, **meta, 'createdAt': now, 'name': name}
        # for k, v in meta.items():
        #     df[k] = v
        self._write_table(**kwargs)

    @selector
    def add_env(self, env):
        filename = os.path.join(self.env_folder, f"{self.meta['mode']}.{self.meta['episode']}.json")
        with open(filename, 'w') as outfile:
            json.dump(env.to_dict(), outfile)


    @selector
    def add_metrics2(self, scope, metrics):
        if scope not in self.traces:
            self.traces[scope] = {
                'values': [], 'episode': [], 'episode_step': [], 'mode': [], 'name': []
            }
        metrics = {k: v.cpu().numpy() for k, v in metrics.items()}
        for k,v in metrics.items():
            self.traces[scope]['episode'].append(self.meta['episode'])
            self.traces[scope]['episode_step'].append(self.meta['episode_step'])
            self.traces[scope]['mode'].append(self.meta['mode'])
            self.traces[scope]['name'].append(k)
            self.traces[scope]['values'].append(v)

        if len(self.traces[scope]['values']) > self.flush_period:
            self.metrics2_flush()

    def metrics2_flush(self):
        for scope_name, traces in self.traces.items():
            values = traces.pop('values')
            index = pd.MultiIndex.from_frame(pd.DataFrame(traces))
            if values[0].size > 1:
                columns = pd.Series([f'agent_{i}' for i in range(len(values[0]))], name='agents')
                df = pd.DataFrame(data=values, index=index, columns=columns)
            else:
                df = pd.DataFrame(data=values, index=index, columns=['value'])
            metrics_file = os.path.join(self.metrics_folder, f"{scope_name}.{self.flush_idx}.parquet")
            df.to_parquet(metrics_file)
        self.traces = {}

        self.flush_idx += 1

    @selector
    def add_metrics(self, name, metrics, meta, tf=[]):
        if self.tensorboard_writer:
            for n in tf:
                self.tensorboard_writer.add_scalar(n, metrics[n], self.step)
        meta = {**self.meta, **meta}
        if name in self.meta_rows:
            assert name in self.metric_rows
            self.meta_rows[name].append(parse_dict(meta))
            self.metric_rows[name].append(parse_dict(metrics))
        else:
            self.meta_rows[name] = [parse_dict(meta)]
            self.metric_rows[name] = [parse_dict(metrics)]

    @selector
    def add_image(self, name, image):
        name = name.format(**self.meta)
        if self.tensorboard_writer:
            self.tensorboard_writer.add_image(name, image, self.step)
        self._write_image(name, image)

    @selector
    def add_video(self, name, video):
        name = name.format(**self.meta)
        if self.tensorboard_writer:
            self.tensorboard_writer.add_video(name, video, self.step, fps=1)
        assert video.shape[0] == 1, 'Multiple videos are not yet supported'
        self._write_video(name, video[0], fps=1)

    @selector
    def add_frame(self, name, callback, flush=False):
        name = name.format(**self.meta)
        if name not in self.frames:
            self.frames[name] = [callback()]
        else:
            self.frames[name].append(callback())
        if flush:
            self.frames_flush()

    def frames_flush(self):
        for name, frames in self.frames.items():
            video = th.cat(frames, dim=1)
            self.add_video(name, video)
        self.frames = {}

    def _write_image(self, name, array):
        ensure_dir(self.image_folder)
        file_name = os.path.join(self.image_folder, f'{name}.{self.step}.png')
        assert array.shape[0] == 1, 'Multiple images are not yet supported'
        imageio.imwrite(file_name, array[0].detach().numpy())

    def _write_video(self, name, array, fps=1):
        ensure_dir(self.video_folder)
        file_name = os.path.join(self.video_folder, f'{name}.{self.step}.mp4')
        array_np = array.transpose(1,3).transpose(1,2).cpu().numpy()
        clip = ImageSequenceClip([f for f in array_np], fps=1)
        clip.write_videofile(file_name)

    def _write_table(self, df, name, sheet):
        df.to_csv(f"{self.df_folder}/{name}.{sheet}.csv")

    def flush(self):
        self.rows_flush()
        self.metrics2_flush()
        self.flush_idx += 1

    def __del__(self):
        # self.json_writer.close()
        self.flush()

    def rows_flush(self):
        names = self.metric_rows.keys()
        for n in names:
            df_metrics = pd.DataFrame.from_records(self.metric_rows[n])
            df_meta = pd.DataFrame.from_records(self.meta_rows[n]).astype('category')
            df = pd.concat([df_meta, df_metrics], axis=1)
            metrics_file = os.path.join(self.metrics_folder, f"{n}.parquet")
            df.to_parquet(metrics_file)

    @selector
    def write_module(self, name, module):
        name = name.format(**self.meta)
        if self.tensorboard_writer:
            for p_name, values in module.named_parameters():
                self.tensorboard_writer.add_histogram(f'{name}.{p_name}', values, self.step)

    def set_details(self, details):
        self.details = details
