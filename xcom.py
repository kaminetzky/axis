from functools import cache
import requests


def parse_attenuation_table(attenuations_text):
  attenuations_list = [float(num) for num in attenuations_text.split()[5:]]
  attenuations_keys = attenuations_list[::2]
  attenuations_values = attenuations_list[1::2]
  return dict(zip(attenuations_keys, attenuations_values))


@cache
def get_total_attenuation(energies_kev, atomic_number=None,
                          element_symbol=None):
  # TODO: Add file cache
  if atomic_number is not None and element_symbol is not None:
    raise TypeError('atomic_number and element_symbol should not be passed '
                    'together')
  if atomic_number is None and element_symbol is None:
    raise TypeError('atomic_number or element_symbol should be passed')

  energies_mev = [str(energy / 1000) for energy in energies_kev]
  energy_min = min(energies_mev)
  energy_max = max(energies_mev)

  url = 'https://physics.nist.gov/cgi-bin/Xcom/data.pl'
  data = {
    'character': 'space',
    'Method': '1',
    'ZNum': atomic_number,
    'ZSym': element_symbol,
    'OutOpt': 'PIC',
    'NumAdd': '1',
    'Energies': ';'.join(map(str, energies_mev)),
    'Output': '',
    'WindowXmin': energy_min,
    'WindowXmax': energy_max,
    'with': 'on',
  }

  response_text = requests.post(url, data=data).text

  if 'Error' in response_text:
    error_text = response_text[108:-5]
    raise RuntimeError(error_text)

  return parse_attenuation_table(response_text)


def get_attenuation_ratio(low_energy, high_energy, atomic_number=None,
                          element_symbol=None):
  attenuations = get_total_attenuation((low_energy, high_energy),
                                       atomic_number,
                                       element_symbol)
  low_energy_attenuation, high_energy_attenuation = attenuations.values()
  return low_energy_attenuation / high_energy_attenuation


def get_transition_ratios(low_energy, high_energy, inorganic_start=11,
                          metal_start=19):
  # inorganic_start and metal_start declare the effective atomic number of
  # the first element that corresponds to that category. The threshold will be
  # (ratio(category_start) + ratio(categry_start - 1)) / 2.

  atomic_numbers = {'organic end': inorganic_start - 1,
                    'inorganic start': inorganic_start,
                    'inorganic end': metal_start - 1,
                    'metal start': metal_start}

  ratios = {key: get_attenuation_ratio(low_energy, high_energy,
            atomic_number=value) for key, value in atomic_numbers.items()}

  organic_inorganic_threshold = (ratios['organic end']
                                   + ratios['inorganic start']) / 2
  inorganic_metal_threshold = (ratios['inorganic end']
                                 + ratios['metal start']) / 2

  return organic_inorganic_threshold, inorganic_metal_threshold
