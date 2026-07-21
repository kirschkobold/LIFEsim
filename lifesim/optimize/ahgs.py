import numpy as np

from lifesim.core.modules import SlopeModule
from lifesim.optimize.observation_queue import ObservationQueue

class AhgsModule(SlopeModule):
    def __init__(self,
                 name: str):
        super().__init__(name=name)

        self.tot_time = 0
        self.star_data = dict()

    def obs_array_star(self, i):
        """Observation-cost array for the star at index ``i`` in ``self._stars``.

        Operates purely on the precomputed NumPy slices set up in ``distribute_time`` (no
        pandas indexing / full-catalog scans), so the cost is O(planets of this star).
        """
        rows = self._star_rows[i]
        sel = (~self._detected[rows]) & self._is_interesting[rows]

        if not np.any(sel):
            if self._char:
                return np.array(((np.inf, np.inf), (np.inf, np.inf)))
            else:
                return np.array((np.inf, np.inf))

        active = rows[sel]
        k = active.size

        # planets with snr_1h == 0 (filtered-out targets) intentionally yield an infinite
        # observation cost, so they are never selected -> silence the divide-by-zero notice
        with np.errstate(divide='ignore', invalid='ignore'):
            obs = (60 * 60 *
                   (self._snr_target ** 2 - self._snr_current[active] ** 2)
                   / self._snr_1h[active] ** 2)

            # included slew time
            # needs to be divided by the number of universes for proper optimization, since the
            # characterization happens 'per universe'
            if self._char:
                obs_char = (60 * 60 *
                            self._snr_char ** 2
                            / self._maxsep_snr_1h[active] ** 2
                            + self._t_slew) / self._num_universe

        if self._char:
            # characterization time is needed for optimization, detection time is needed to
            # understand how much mission time was used for one observation
            met = np.stack((obs, obs_char), axis=0)
            met = met[:, np.argsort(met[0])]
            met[1] = np.cumsum(met[1])
            # t_slew is constant across a star's (undetected) planets, so the ordering mismatch
            # with the argsort above is irrelevant
            met[0] -= self._t_slew_arr[active]
            met[1] += met[0]
            met = met / np.arange(1, k + 1, 1)[np.newaxis, :]

            return met
        else:
            obs -= self._t_slew_arr[active]
            obs = np.sort(obs) / np.arange(1, k + 1, 1)

            return obs

    def observe_star(self,
                     i,
                     int_time,
                     delete=False):

        if delete:
            raise ValueError('Delete mode not implemented for AHGS optimizer.')

        rows = self._star_rows[i]

        self.data.optm['tot_time'] += int_time
        if i not in self.star_data:
            r0 = rows[0]
            data = {'t_slew': self._t_slew_arr[r0],
                    'int_time': self._int_time_arr[r0]}
            self.star_data[i] = data
        else:
            data = self.star_data[i]

        slew_time = data['t_slew']
        if not (slew_time == 0):
            if (slew_time + int_time) < 0:
                data['t_slew'] += int_time
                int_actual = 0
            else:
                data['t_slew'] = 0
                data['int_time'] += (slew_time + int_time)
                int_actual = slew_time + int_time
        else:
            data['int_time'] += int_time
            int_actual = int_time

        self._snr_current[rows] = np.sqrt(
            self._snr_current[rows] ** 2
            + (self._snr_1h[rows] * np.sqrt(int_actual / (60 * 60))) ** 2)

        # Check if planets of this star have been detected for the first time. Only this star's
        # snr_current changed, so only its rows can newly cross the threshold.
        newly = rows[(~self._detected[rows])
                     & (self._snr_current[rows] >= self._snr_target)]

        if newly.size > 0:
            self._detected[newly] = True
            self._t_detected[newly] = self.tot_time + int_time
            self._t_slew_arr[newly] = data['t_slew']
            self._int_time_arr[newly] = data['int_time']

            # vectorized experiment / per-universe bookkeeping over the (few) new detections
            for j, exp in enumerate(self._exp_names):
                members = newly[self._exp_matrix[newly, j]]
                if members.size > 0:
                    self.data.optm['exp_detected'][exp] += int(members.size)
                    uni_row = self.data.optm['exp_detected_uni'][exp]
                    idx = np.searchsorted(uni_row[0], self._nuniverse[members])
                    np.add.at(uni_row[1], idx, 1)

    def reset_queue(self, stars, maximum_occurrence):

        observation_queue = ObservationQueue()
        if self._char:
            # fill the observation time array
            for i in range(len(stars)):
                temp = self.obs_array_star(i)
                stop_index = temp.shape[1]
                for ind_star in range(maximum_occurrence):
                    val_det, val_char = temp[:, ind_star] if ind_star < stop_index else (np.inf, np.inf)
                    observation_queue.put((val_char, (i, ind_star, val_det)))
        else:
            # fill the observation time array
            for i in range(len(stars)):
                temp = self.obs_array_star(i)
                stop_index = temp.shape[0]
                for ind_star in range(maximum_occurrence):
                    value = temp[ind_star] if ind_star < stop_index else np.inf
                    observation_queue.put((value, (i, ind_star)))

        return observation_queue

    def distribute_time(self):
        cat = self.data.catalog

        # ------------------------------------------------------------------
        #  Set up NumPy working arrays and per-star row groups (done once).
        #  The greedy loop below then never touches pandas.
        # ------------------------------------------------------------------
        nstar_arr = cat['nstar'].to_numpy()
        order = np.argsort(nstar_arr, kind='stable')
        stars, start, counts = np.unique(nstar_arr[order],
                                         return_index=True, return_counts=True)
        # row positions (into the catalog arrays) for each star; index matches `stars`
        self._stars = stars
        self._star_rows = [order[start[i]:start[i] + counts[i]] for i in range(len(stars))]

        # cached optimization constants
        opt = self.data.options.optimization
        self._char = opt['characterization']
        self._snr_target = opt['snr_target']
        self._snr_char = opt['snr_char']
        self._t_slew = self.data.options.array['t_slew']
        self._num_universe = self.data.optm['num_universe']

        # immutable lookups
        self._snr_1h = cat['snr_1h'].to_numpy().astype(float)
        self._maxsep_snr_1h = (cat['maxsep_snr_1h'].to_numpy().astype(float)
                               if self._char else None)
        self._nuniverse = cat['nuniverse'].to_numpy()
        self._habitable = cat['habitable'].to_numpy()
        self._exp_names = [c[4:] for c in cat.columns if c.startswith('exp_')]
        self._exp_matrix = (cat[['exp_' + e for e in self._exp_names]].to_numpy()
                            if self._exp_names else np.zeros((len(nstar_arr), 0), dtype=bool))

        # mutable state (written back to the catalog at the end)
        self._snr_current = cat['snr_current'].to_numpy().astype(float).copy()
        self._detected = cat['detected'].to_numpy().astype(bool).copy()
        self._is_interesting = cat['is_interesting'].to_numpy().astype(bool).copy()
        self._t_slew_arr = cat['t_slew'].to_numpy().astype(float).copy()
        self._int_time_arr = cat['int_time'].to_numpy().astype(float).copy()
        self._t_detected = cat['t_detected'].to_numpy().astype(float).copy()

        self.star_data = dict()

        maximum_occurrence = int(np.max(counts))
        observation_queue = self.reset_queue(stars, maximum_occurrence)

        obs_time = (opt['t_search'] * self.data.options.array['t_efficiency'])

        self.tot_time = 0

        print('Number of planets detected for each experiment:')

        def status_string():
            out = ''
            for key in self.data.optm['exp_detected'].keys():
                out += (key + ': '
                        + str(self.data.optm['exp_detected'][key] / self._num_universe)
                        + '  ')
            if opt['opt_limit'] == 'experiments':
                out += ('-  ' + str(np.round(self.tot_time / 60 / 60 / 24 / 365.25, decimals=1))
                        + ' yrs observed')
            else:
                out += ('-  (' + str(np.round(self.tot_time / 60 / 60 / 24 / 365.25, decimals=1)) + ' / '
                        + str(np.round(obs_time / 60 / 60 / 24 / 365.25, decimals=1))
                        + ') yrs observed')
            return out

        run_bool = True
        iter_count = 0

        while run_bool:
            if self._char:
                # find the best global slope and observe star
                val_char, (no_star, ind_t, val_det) = observation_queue.get()
                if not np.isfinite(val_char):
                    print('Not sufficient targets remaining to continue characterization optimization.')
                    break
                if (((self.tot_time + val_det * (ind_t + 1) + 0.01) > obs_time)
                        and (opt['opt_limit'] == 'time')):
                    rem_time = obs_time - self.tot_time
                    self.observe_star(i=no_star,
                                      int_time=rem_time)
                    self.tot_time += rem_time
                else:
                    if ((val_char - val_det) * (ind_t + 1) + 0.01) < 0:
                        raise ValueError('Negative time difference encountered.')
                    self.observe_star(i=no_star,
                                      int_time=val_det * (ind_t + 1) + 0.01)
                    temp = self.obs_array_star(no_star)
                    self.tot_time += val_det * (ind_t + 1) + 0.01
                    stop_index = temp.shape[1]
                    for ind_star in range(maximum_occurrence):
                        val_det, val_char = temp[:, ind_star] if ind_star < stop_index else (np.inf, np.inf)
                        observation_queue.put((val_char, (no_star, ind_star, val_det)))
            else:
                value, (no_star, ind_t) = observation_queue.get()
                if (((self.tot_time + value * (ind_t + 1) + 0.01) > obs_time)
                        and (opt['opt_limit'] == 'time')):
                    rem_time = obs_time - self.tot_time
                    self.observe_star(i=no_star,
                                      int_time=rem_time)
                    self.tot_time += rem_time
                else:
                    self.observe_star(i=no_star,
                                      int_time=value * (ind_t + 1) + 0.01)
                    temp = self.obs_array_star(no_star)
                    self.tot_time += value * (ind_t + 1) + 0.01
                    stop_index = temp.shape[0]
                    for ind_star in range(maximum_occurrence):
                        value = temp[ind_star] if ind_star < stop_index else np.inf
                        observation_queue.put((value, (no_star, ind_star)))

            # status line is purely cosmetic -> throttle it to avoid per-iteration string/IO cost
            iter_count += 1
            if iter_count % 200 == 0:
                print('\r' + status_string(), end='')

            if any([
                ((self.data.optm['exp_detected_uni'][exp][1, :]
                 > opt['experiments'][exp]['sample_size']).sum() >
                (opt['opt_limit_factor']
                        * self._num_universe)) and not self.data.optm['hit_limit'][exp]
                for exp in self.data.optm['exp_detected_uni']]):
                over_limit_experiments = [
                    exp for exp in self.data.optm['exp_detected_uni']
                    if
                    (self.data.optm['exp_detected_uni'][exp][1, :]
                     > opt['experiments'][exp]['sample_size']).sum() >
                    (opt['opt_limit_factor']
                            * self._num_universe)
                    and not self.data.optm['hit_limit'][exp]
                ]

                for exp in over_limit_experiments:
                    self.data.optm['hit_limit'][exp] = True
                    self._is_interesting[:] = False
                    for exp_interesting in [e for e, hit in self.data.optm['hit_limit'].items() if not hit]:
                        j = self._exp_names.index(exp_interesting)
                        self._is_interesting |= self._exp_matrix[:, j]

                if self._is_interesting.sum() == 0:
                    if opt['opt_limit'] == 'time':
                        print('\nAll experiments have been completed, spending remaining mission time on all HZ planets.')
                    self._is_interesting = self._habitable.astype(bool).copy()

                else:
                    print('\nCompleted experiments: ' + ', '.join(over_limit_experiments)
                          + ', RECOUNTING -------------------')

                # fill the observation time array
                if len(over_limit_experiments) > 1:
                    observation_queue = self.reset_queue(stars, maximum_occurrence)

            if opt['opt_limit'] == 'time':
                run_bool = self.tot_time < obs_time
            elif opt['opt_limit'] == 'experiments':
                max_experiment_time = 50 * 365.25 * 24 * 60 * 60
                run_bool = (not all(self.data.optm['hit_limit'].values()) and self.tot_time < max_experiment_time)
            else:
                raise ValueError('Optimization limit not recognized.')

        # final status update (the throttled loop may have skipped the last few iterations)
        print('\r' + status_string(), end='')

        # ------------------------------------------------------------------
        #  Write the mutated state back into the catalog (positional assignment).
        # ------------------------------------------------------------------
        cat['snr_current'] = self._snr_current
        cat['detected'] = self._detected
        cat['t_detected'] = self._t_detected
        cat['t_slew'] = self._t_slew_arr
        cat['int_time'] = self._int_time_arr
        cat['is_interesting'] = self._is_interesting