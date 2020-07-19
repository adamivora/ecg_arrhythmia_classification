from os import path

import neurokit2 as nk
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wfdb
from scipy.signal import resample_poly

from detection.preprocessing.dataset import Cinc2017Dataset, Cpsc2018Dataset
from detection.utils.filesystem import ensure_directory_exists, images_dir


class ImageGenerator:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_dir = images_dir(root_dir)

    def set_ecg_layout(self, fig, font=None, **kwargs):
        if font is None:
            font = dict(
                family='sans serif',
                size=12,
            )
        fig.update_layout(
            xaxis_title='time (s)',
            yaxis_title='voltage (μV)',
            font=font,
            **kwargs
        )

    def plot_signal(self, signal, fs, seconds=None, title=''):
        if seconds is not None:
            signal = signal[:fs * seconds + 1]
        fig = px.line(x=np.arange(len(signal)) / fs, y=signal)
        self.set_ecg_layout(fig, title=title)
        return fig

    def plot_row(self, dataset, row):
        signal = dataset.read_record(row.Record)
        self.plot_signal(signal, row.Fs, f'{row.Record} - {row.Label}')

    def plot_row_resampled(self, dataset, row, fs, font=None):
        signal = dataset.read_record(row.Record)
        resampled = resample_poly(signal, fs, row.Fs)
        fig = self.plot_signal(resampled, fs, 10, f'{row.Record} ({fs} Hz)')
        fig.update_layout(font=font)
        return fig

    def plot_at_sample_rates(self, dataset, row, sample_rates, show=True, save=False):
        figures = [
            self.plot_row_resampled(
                dataset, row, fs,
                font=dict(
                    family='sans serif',
                    size=24,
                ))
            for fs in sample_rates
        ]

        if show:
            for fig in figures:
                fig.show()
        if save:
            for fig, fs in zip(figures, sample_rates):
                self.save_image(fig, f'ecg_{fs}Hz.png', width=1800, height=300)

    def save_image(self, fig, name, width=900, height=300, **kwargs):
        ensure_directory_exists(self.image_dir)
        fig.write_image(path.join(self.image_dir, name), width=width, height=height, **kwargs)

    def sample_rate_comparison(self, dataset, row):
        self.plot_at_sample_rates(dataset, row, [300, 100, 50, 10, 1], show=False, save=True)

    def example_ecg(self, dataset):
        normal = dataset.data.iloc[85]
        afib = dataset.data.iloc[71]
        noise = dataset.data[dataset.data.Label == '~'].iloc[6]

        self.save_image(self.plot_signal(dataset.read_record(normal.Record), normal.Fs, seconds=5,
                                         title=f'{normal.Record} - Normal'),
                        'normal_ecg.png', width=900, height=300)
        self.save_image(self.plot_signal(-dataset.read_record(afib.Record)[500:], afib.Fs, seconds=5,
                                         title=f'{afib.Record} - Atrial fibrillation'),
                        'afib_ecg.png', width=900, height=300)
        self.save_image(
            self.plot_signal(dataset.read_record(noise.Record), noise.Fs, seconds=5, title=f'{noise.Record} - Noise'),
            'noise_ecg.png',
            width=900, height=300)

    def qrs_detection_pantompkins_vs_neurokit(self, dataset):
        row = dataset.data.iloc[50]
        signal = dataset.read_record(row.Record)[:row.Fs * 20 + 1]

        method_names = {
            'pantompkins': 'Pan–Tompkins',
            'neurokit': 'Neurokit'
        }

        for method in ['pantompkins', 'neurokit']:
            ecg_cleaned = nk.ecg_clean(signal, sampling_rate=row.Fs, method=method)
            peaks, info = nk.ecg_peaks(ecg_cleaned, sampling_rate=row.Fs, method=method)

            r_peaks = np.where(peaks['ECG_R_Peaks'].to_numpy() == 1)[0]

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=np.arange(len(signal)) / row.Fs, y=signal))
            fig.add_trace(go.Scatter(mode='markers', x=r_peaks / row.Fs, y=signal[r_peaks]))
            fig.update_traces(marker=dict(size=8))
            self.set_ecg_layout(fig, title=f'{row.Record} - R peaks ({method_names[method]} method)', showlegend=False,
                                xaxis=dict(range=[0, 20]), yaxis=dict(range=[-5000, 5000]))
            self.save_image(fig, f'qrs_{method}.png', width=900, height=300)

    def pvc_vt_images(self, dataset):
        pvc_signal = dataset.read_record('A0050') * 1e3

        vt_record = wfdb.rdrecord('cu05', pn_dir='cudb/1.0.0/')
        vt_signal = vt_record.p_signal.reshape(-1)[93200:] * 1e3

        self.save_image(
            self.plot_signal(pvc_signal, fs=500, title='Premature ventricular contraction', seconds=10),
            'pvc.png', width=900, height=300)
        self.save_image(self.plot_signal(vt_signal, fs=250, title='Ventricular tachycardia', seconds=10),
                        'vt.png', width=900, height=300)

    def generate_images(self):
        """
        Generate all the plots and figures for the thesis `ECG Arrhythmia Detection and Classification`.
        """

        print('Generating all plots and images...')
        dataset = Cinc2017Dataset(root_dir=self.root_dir, split='trainval')
        dataset_train = Cinc2017Dataset(root_dir=self.root_dir, split='train')
        cpsc_dataset = Cpsc2018Dataset(root_dir=self.root_dir)

        self.example_ecg(dataset_train)
        self.pvc_vt_images(cpsc_dataset)
        self.sample_rate_comparison(dataset_train, dataset_train.data.iloc[0])
        self.qrs_detection_pantompkins_vs_neurokit(dataset)

        print(f'Generated all plots and images to {self.image_dir}.')
