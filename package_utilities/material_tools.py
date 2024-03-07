from mendeleev.fetch import fetch_table
import os
import sys
import numpy as np
import copy
import pandas as pd

class Isotopic_Database:
    """
    This class creates a database with the isotopic properties.
    """
    def __init__(self):
        self.__isotope_dataframe = {}
        self.__natural_elements = {}
        iso_df = fetch_table('isotopes', index_col='id')
        ele_df = fetch_table('elements')
        cols = ['symbol','atomic_number']
        merged = pd.merge(ele_df[cols], iso_df, how='outer', on='atomic_number')
        for iso in range(len(merged['symbol'])):
            key_iso = merged['symbol'][iso]+'-'+str(merged['mass_number'][iso])
            key_zai = str(merged['atomic_number'][iso]*1000+merged['mass_number'][iso])
            key_zai_nat = str(merged['atomic_number'][iso]*1000)
            if key_zai_nat not in self.__natural_elements.keys():
                self.__natural_elements[key_zai_nat] = {'isotopes':[],'at':[],'symbol':[]}
            self.__isotope_dataframe[key_zai] = {'mass':merged['mass'][iso],
                                                 'symbol':key_iso}
            
            if np.isnan(merged['abundance'][iso]) == True:
                self.__isotope_dataframe[key_zai]['abundance'] = 0.0
            else:
                self.__isotope_dataframe[key_zai]['abundance'] = merged['abundance'][iso]/100
                self.__natural_elements[key_zai_nat]['isotopes'].append(key_zai)
                self.__natural_elements[key_zai_nat]['symbol'].append(key_iso)
                self.__natural_elements[key_zai_nat]['at'].append(merged['abundance'][iso]/100)
        
        
        # --- # APPLICATION OF CORRECTIONS ON NUCLEAR DATA # --- #
        
        self.__isotope_dataframe['1001']['abundance'] = 0.999885
        self.__isotope_dataframe['1002']['abundance'] = 0.000115
        self.__natural_elements['1000']['at'][1] = self.__isotope_dataframe['1002']['abundance']
        self.__natural_elements['1000']['at'][0] = self.__isotope_dataframe['1001']['abundance']
        
        self.__isotope_dataframe['73180']['abundance'] = 0.0001201
        self.__isotope_dataframe['73181']['abundance'] = 0.9998799
        self.__isotope_dataframe['73180']['mass'] = 179.9474648
        self.__isotope_dataframe['73181']['mass'] = 180.9479958
        self.__isotope_dataframe['73180']['symbol'] = 'Ta-180'
        self.__isotope_dataframe['73181']['symbol'] = 'Ta-181'
        self.__natural_elements['73000']['isotopes'] = ['73180','73181']
        self.__natural_elements['73000']['symbol'] = ['Ta-180','Ta-181']
        self.__natural_elements['73000']['at'] = [self.__isotope_dataframe['73180']['abundance'],self.__isotope_dataframe['73181']['abundance']]
        
        self.__isotope_dataframe['16032']['abundance'] = 0.9499
        self.__isotope_dataframe['16033']['abundance'] = 0.0075
        self.__isotope_dataframe['16034']['abundance'] = 0.0425
        self.__isotope_dataframe['16036']['abundance'] = 0.0001
        self.__natural_elements['16000']['at'] = [self.__isotope_dataframe['16032']['abundance'],self.__isotope_dataframe['16033']['abundance'],self.__isotope_dataframe['16034']['abundance'],self.__isotope_dataframe['16036']['abundance']]
        

    def get_natural_elements(self):
        """
        This method extract the isotopes which have a natural composition.
        """
        results = copy.deepcopy(self.__natural_elements)
        return results   
    
    def get_output(self):
        """
        This function returns the the database
        """
                    
        results = self.__isotope_dataframe
        return results
                    
class Conversion_wt_at:
    """
    This class stores the functions to convert a composition from isotopic fraction to atomic fraction and viceversa.
    It also evaluates the effective atomic mass of the composition.
    """
    def __init__(self):
        ID = Isotopic_Database()
        self.__iso_database = ID.get_output()
        self.__nat_elements = ID.get_natural_elements()
    
    def wt_to_at(self, list_isotopes, list_comp):
        """
        This function performs the conversion from weight fraction to isotopic fraction.
        
        Args:
        list_isotopes: list
            List of the isotopes ZAI in string forms.
        list_comp: list
            List of float expressing the composition in weight fraction.
            
        """
        norm_factor = sum([comp*10**15 for comp in list_comp])
        mod_comp = [comp*10**15 for comp in list_comp]
        mass = (sum([mod_comp[iso]/self.__iso_database[list_isotopes[iso]]['mass'] for iso in range(len(list_isotopes))])/norm_factor)**(-1)
        at_comp = [mod_comp[iso]*mass/(self.__iso_database[list_isotopes[iso]]['mass']*norm_factor) for iso in range(len(list_isotopes))]
        wt_comp = [comp/norm_factor for comp in mod_comp]
        return at_comp, wt_comp, mass

    def at_to_wt(self, list_isotopes, list_comp):
        """
        This function performs the conversion from isotopic fraction to weight fraction.
        
        Args:
        list_isotopes: list
            List of the isotopes ZAI in string forms.
        list_comp: list
            List of float expressing the composition in isotopic fraction.
            
        """
        norm_factor = sum([comp*10**15 for comp in list_comp])
        mod_comp = [comp*10**15 for comp in list_comp]
        mass = sum([mod_comp[iso]*self.__iso_database[list_isotopes[iso]]['mass'] for iso in range(len(list_isotopes))])/norm_factor
        wt_comp = [mod_comp[iso]*self.__iso_database[list_isotopes[iso]]['mass']/(mass*norm_factor) for iso in range(len(list_isotopes))]
        at_comp = [comp/norm_factor for comp in mod_comp]
        return at_comp, wt_comp, mass

    def mixer(self, _dict_vectors, _comp_mixture):
        """
        This function will perform the mixing between composition and return the mixing compositions.

        Args:
            _dict_vectors: dict
                dict_vectors.keys(): is the name of the vector
                _dict_vector[name_vector]: dict;
                    _dict_vector[name_vector].keys(): 'isotopes','at','wt','mass'. 
                        _dict_vector[name_vector]['isotopes']: list;
                            List of ZAI isotopic identifiers
                        _dict_vector[name_vector]['symbol']: list;
                            List of isotope symbols
                        _dict_vector[name_vector]['wt']: list;
                            List of weight fraction.
                        _dict_vector[name_vector]['at']: list;
                            List of isotopic fraction.
                        _dict_vector[name_vector]['mass']: float;
                            Atomic mass of the composition
            _comp_mixture: dict
                _comp_mixture.keys(): 'vectors', 'composition'
                    _comp_mixture['vectors']: list;
                        List of the names of the vectors to be inserted in the mixture. 
                    _comp_mixture['composition']: list;
                        List of weight fraction (negative values) or isotopic fraction (positive values)
                        of the single components inside the mixture
        
        """
        # --- #  CHECK ON INPUT DATA # --- #
        list_check = ["p" if comp > 0 else "n" if comp < 0 else "z" for comp in _comp_mixture['composition']]
        if 'p' in list_check and 'n' in list_check:
                raise ValueError('Error! Unable to evaluate the composition if both isotopic and mass fraction are present.')
        elif all(c=='z' for c in list_check):
                raise ValueError('Error! The composition is made only by null values.')
        for vect in _comp_mixture['vectors']:
            if vect not in _dict_vectors.keys():
                raise NameError('Error! Vector not found.')
        # --- # PERFORMING THE MIXING # --- #
        norm_factor_mix = sum([comp*10**15 for comp in _comp_mixture['composition']])
        final_comp_mix = [comp*10**15/norm_factor_mix for comp in _comp_mixture['composition']]
        mix_dict = {}
        isotopes_mix = []
        symbol_mix = []
        comp_mix = []
        for vect in range(len(_comp_mixture['vectors'])):
            isotopes_mix = isotopes_mix + _dict_vectors[_comp_mixture['vectors'][vect]]['isotopes']
            symbol_mix = symbol_mix + _dict_vectors[_comp_mixture['vectors'][vect]]['symbol']
            if 'p' in list_check and 'n' not in list_check:
                comp_mix = comp_mix + [final_comp_mix[vect]*_dict_vectors[_comp_mixture['vectors'][vect]]['at'][iso]
                                       for iso in range(len(_dict_vectors[_comp_mixture['vectors'][vect]]['isotopes']))]
            elif 'n' in list_check and 'p' not in list_check:
                comp_mix = comp_mix + [final_comp_mix[vect]*_dict_vectors[_comp_mixture['vectors'][vect]]['wt'][iso]
                                       for iso in range(len(_dict_vectors[_comp_mixture['vectors'][vect]]['isotopes']))]
        
        for iso in range(len(isotopes_mix)):
            if isotopes_mix[iso] not in mix_dict.keys():
                mix_dict[isotopes_mix[iso]] = [0.0,symbol_mix[iso]]
            mix_dict[isotopes_mix[iso]][0] = mix_dict[isotopes_mix[iso]][0] + comp_mix[iso]
        
        res_comp = []
        res_dict = {'isotopes':[],'symbol':[]}
        for ikey, ires in mix_dict.items():
            
            res_dict['isotopes'].append(ikey)
            res_dict['symbol'].append(ires[1])
            res_comp.append(ires[0])
            
        if 'p' in list_check and 'n' not in list_check:
            res_dict['at'], res_dict['wt'], res_dict['mass'] = self.at_to_wt(res_dict['isotopes'],res_comp)
        elif 'n' in list_check and 'p' not in list_check:
            res_dict['at'], res_dict['wt'], res_dict['mass'] = self.wt_to_at(res_dict['isotopes'],res_comp)
        
        return res_dict
        
    def get_database(self):
        """
        This function return the database.
        """
        iso_db =  copy.deepcopy(self.__iso_database)
        nat_ele_db = copy.deepcopy(self.__nat_elements)
        return iso_db, nat_ele_db

class Vector_Generator:
    """
    This class creates the composition of a generic fuel.
    Args:
        oxigen_comp: 'nat' or dict
            If oxigen_comp is a dictionary:
                oxigen_comp.keys(): ZAI
                oxigen_comp.items(): float
                    Positive value for isotopic fractions and negative values for mass fraction.
    """    
    def __init__(self):
        self.__vectors = {}
        self.__CWA = Conversion_wt_at()
        self.__ID, self.__NE = self.__CWA.get_database()
    
    def get_database(self):
        """
        This function return the isotopic database and natural elements database.
        """
        iso_db = copy.deepcopy(self.__ID)
        nat_ele_db = copy.deepcopy(self.__NE)
        return iso_db, nat_ele_db
    
    def add_generic_vector(self, name_vector, comp_vector, extra_in_vect = {}):
        """
        This function will add vectors to the classes to be used for the creation of the composition.

        Args:
            name_vector: str
                Name of the vector
            comp_vector: dict
                comp_vector.keys(): str:
                    ZAI isotopic identifiers. 
                    If ZAI identifiers is in the form "ZZ000", then the natural composition, if present, is extracted.
                comp_vector.values(): float
                    Mass fraction (negative values) or Isotopic fraction (positive value) of the composition.
                    All the values must have the same sign. Positive and negative values are not accepted.
                    Not normamlized compositions are accepted.
                    
            extra_in_vect: dict, optional;
                extra_in_vect.keys(): str;
                    ZAI isotopic identifier
                extra_in_vect.values(): float in (-1, 1);
                    Mass fraction (negative values) or Isotopic fraction (positive values) of the isotope into the vector.
                    All the composition must be expressed with the same methodology,
                    so no positive and negative values are accepted at the same time.
                    They must be expressed as:
                        atoms_extra/(atoms_vect+atoms_extra)
                    or:
                        kg_extra/(kg_vect+kg_extra)
        """
        
        # --- # EXTRACTING COMPOSITION FROM THE comp_vector DICTIONARY # --- #
        list_vect_isotopes = []
        list_vect_comp = []
        list_vect_symbol = []
        dict_vect = {}
        if isinstance(comp_vector, dict):
            for key, res in comp_vector.items():
                if key in self.__ID.keys():
                    list_vect_isotopes.append(key)
                    list_vect_comp.append(res)
                    list_vect_symbol.append(self.__ID[key]['symbol'])
                else:
                    raise NameError('Error! The isotope selected is not in the database.')
        elif isinstance(comp_vector, str)  and comp_vector in self.__NE.keys():
            list_vect_isotopes = self.__NE[comp_vector]['isotopes']
            list_vect_symbol = self.__NE[comp_vector]['symbol']
            list_vect_comp = self.__NE[comp_vector]['at']
        else:
             raise NameError('Error! Unable to identify the element/isotope.')
         
        list_check = ["p" if comp > 0 else "n" if comp < 0 else "z" for comp in list_vect_comp]
        if 'p' in list_check and 'n' in list_check:
            raise ValueError('Error! Unable to evaluate the composition if both isotopic and mass fraction are present.')
        elif 'p' in list_check and 'n' not in list_check:
            dict_vect['isotopes'] = list_vect_isotopes
            dict_vect['at'], dict_vect['wt'], dict_vect['mass'], = self.__CWA.at_to_wt(dict_vect['isotopes'], list_vect_comp)
        elif 'n' in list_check and 'p' not in list_check:
            dict_vect['isotopes'] = list_vect_isotopes
            dict_vect['at'], dict_vect['wt'], dict_vect['mass'], = self.__CWA.wt_to_at(dict_vect['isotopes'], list_vect_comp)
        dict_vect['symbol'] = list_vect_symbol
        # --- # EXTRACTING ISOTOPES FROM MA_in_vect # --- #
        if len(extra_in_vect.keys()) > 0:
            list_ma_isotopes = []
            list_ma_comp = []
            list_ma_symbol = []
            for key, res in extra_in_vect.items():
                if key in self.__ID.keys() and key not in dict_vect['isotopes']:
                    list_ma_isotopes.append(key)
                    list_ma_symbol.append(self.__ID[key]['symbol'])
                    list_ma_comp.append(res)
                elif key not in self.__ID.keys():
                    raise NameError('Error! The isotope selected is not in the database.')
            fraction_ma = np.abs(sum(list_ma_comp))
            list_check = ["p" if comp > 0 else "n" if comp < 0 else "z" for comp in list_ma_comp]
            if 'p' in list_check and 'n' in list_check:
                raise ValueError('Error! Unable to evaluate the composition if both isotopic and mass fraction are present.')
            elif all(c=='z' for c in list_check):
                ma_comp_type = 'at'
            else:
                if fraction_ma >= 1.0:
                    raise ValueError('Error! Unable to construct the composition because the fraction of MA is equal or grater to 1.0.')
                if 'p' in list_check and 'n' not in list_check:
                    ma_comp_type = 'at'
                elif 'n' in list_check and 'p' not in list_check:
                    ma_comp_type = 'wt'
                
                    
            final_dict_vect = {'isotopes': dict_vect['isotopes']+list_ma_isotopes,'symbol':dict_vect['symbol']+list_ma_symbol}
            final_list_vect_comp = []
            for i_vect in range(len(list_vect_isotopes)):
                final_list_vect_comp.append(dict_vect[ma_comp_type][i_vect]*(1-fraction_ma))
            for i_ma in range(len(list_ma_isotopes)):
                final_list_vect_comp.append(list_ma_comp[i_ma])
            if ma_comp_type == 'wt':
                final_dict_vect['at'], final_dict_vect['wt'], final_dict_vect['mass'] = self.__CWA.wt_to_at(final_dict_vect['isotopes'],final_list_vect_comp)
            elif ma_comp_type == 'at':
                final_dict_vect['at'], final_dict_vect['wt'], final_dict_vect['mass'] = self.__CWA.at_to_wt(final_dict_vect['isotopes'],final_list_vect_comp)
        else:
            final_dict_vect = copy.deepcopy(dict_vect)
        self.__vectors[name_vector] = copy.deepcopy(final_dict_vect)
    
    def add_natural_vector(self, name_vector, element):
        """
        This function will add a vector with a natural composition.

        Args:
            name_vector: str;
                Name of the vector to be created
            element: str;
                ZAI of the element, to be written in the ZZ000 form. 

        """
        self.add_generic_vector(name_vector,element)
    
    def mixer_vectors(self, name_vector, list_vectors, list_composition):
        """
        This function will mix the vectors reported into the list_vectors for a given composition to create a new vector.
        
        Args:
            name_vector: str
                Name of the vector to be constructed.
            list_vectors: list
                List of the vectors to be mixed
            list_composition: list;
                list_composition[i]: float
                List of composition. Positive value for isotopic fraction and negative values for mass fractions.

        """
        dict_mix = {}
        for vect in list_vectors:
            if vect not in self.__vectors.keys():
                raise NameError('Error! The vector selected not present in the database.')
            else:
                dict_mix[vect] = copy.deepcopy(self.__vectors[vect])
        list_check = ["p" if comp > 0 else "n" if comp < 0 else "z" for comp in list_composition]
        if 'p' in list_check and 'n' in list_check:
                raise ValueError('Error! Unable to evaluate the composition if both isotopic and mass fraction are present.')
        elif all(c=='z' for c in list_check):
                raise ValueError('Error! The composition is made only by null values.')
        self.__vectors[name_vector] = self.__CWA.mixer(dict_mix,{'vectors':list_vectors,'composition':list_composition})

    def get_vectors(self):
        """
        This function will return the vector dictionary.
        """
        results = copy.deepcopy(self.__vectors)
        return results
        

class Material_Generator:
    """
    This class will create the composition of a general material.
    
    """
    def __init__(self):
        self.__VG = Vector_Generator()
        self.__iso_db, self.__ele_db = self.__VG.get_database()
        self.__list_vectors = []
    
    def generic_vector(self, name_vector, comp_vector):
        """
        This function will add a vector witha composition different from the natural one.

        Args:
            name_vector: str
                Name of the vector
            comp_vector: dict
                comp_vector.keys(): str:
                    ZAI isotopic identifiers. 
                    If ZAI identifiers is in the form "ZZ000", then the natural composition, if present, is extracted.
                comp_vector.values(): float
                    Mass fraction (negative values) or Isotopic fraction (positive value) of the composition.
                    All the values must have the same sign. Positive and negative values are not accepted.
                    Not normamlized compositions are accepted.
        """
        self.__VG.add_generic_vector(name_vector,comp_vector)
        if name_vector not in self.__list_vectors:
            self.__list_vectors.append(name_vector)
    
    def material_creator(self, list_vectors, list_composition):
        """
        This class creates materials made by elements with natural composition and 
        vectors created with the method generic_vector().
        
        Args:
            list_element: list;
                list_element[i]: str;
                List of ZAI of natural elements or name of vectors vcreated with generic_vector(). Example '8000' for Oxigen, '92000' for Uranium and so on.
            list_composition: list;
                list_composition[i]: float
                List of composition. Positive value for isotopic fraction and negative values for mass fractions.
        """
        list_check = ["p" if comp > 0 else "n" if comp < 0 else "z" for comp in list_composition]
        if 'p' in list_check and 'n' in list_check:
                raise ValueError('Error! Unable to evaluate the composition if both isotopic and mass fraction are present.')
        elif all(c=='z' for c in list_check):
                raise ValueError('Error! The composition is made only by null values.')
        
        for ele in list_vectors:
            if ele in self.__ele_db.keys() and ele not in self.__list_vectors:
                self.__VG.add_natural_vector(ele, ele)
                self.__list_vectors.append(ele)
            elif ele not in self.__ele_db.keys() and ele not in self.__list_vectors:
                raise NameError('Error! The element does not exist in the database.')
            
        self.__VG.mixer_vectors('result', list_vectors, list_composition)
        results = self.__VG.get_vectors()
        return results['result']

class Fuel_Generator:
    
    def __init__(self):
        """
        
        """
        wip = 'wip'
    
    def uranium_vector(self, comp_vector, dict_MA):
        """_summary_

        Args:
            comp_uranium (_type_): _description_
        """
        wip = 'wip'
    
    def oxigen_vector(self, comp_vector):
        """
        
        Args:
            comp_oxigen (_type_): _description_
        """
        wip = 'wip'
    
    def plutonium_vector(self, comp_vector, dict_MA):
        """
        

        Args:
            comp_vector (_type_): _description_
        """
    
    def burnable_absorber(self,name_vector, comp_vector, dict_MA):
        """_summary_

        Args:
            name_vector (_type_): _description_
            comp_vector (_type_): _description_
            dict_MA (_type_): _description_
        """
                    
    def uo2_calculator(self):
        """
        This class will conctruct the fuel from the vectors constructed with the method add_vector().
        Args:
        
        """
        wip = 'wip'
        
    def mox_density_calculator(self, o_hm_ratio):
        """
        This function will evaluate the density of the fuel.
        

        Args:
            o_hm_ratio (_type_): _description_
        """
        
        def uo2_lat_dim(o_hm_ratio):
            if o_hm_ratio < 2:
                par = 0.0175
            elif o_hm_ratio > 2:
                par = 0.0940
            else:
                par = 0.0
            lat_dim = 5.4704 + par*(2-o_hm_ratio)
            return lat_dim

        def puo2_lat_dim(o_hm_ratio):
            if o_hm_ratio < 2:
                lat_dim = 6.1503 - 0.3789*o_hm_ratio
            elif o_hm_ratio >= 2 and o_hm_ratio <= 2.005:
                lat_dim = 3.92872 + 0.73364*o_hm_ratio
            else:
                lat_dim = 5.3643 + 0.01764*o_hm_ratio
            return lat_dim
        
    def uo2_gd_density_calculator(self, gd_wt_fraction):
        """
        This function will evaluate the density of (U-Gd)O_2 mixture and return its value in g/cm3.
        The function uses a simplified model reported on the paper [1].
        
        [1] A. V. Fedotov, E. N. Mikheev, A. V. Lysikov, and V. V. Novikov, THEORETICAL AND EXPERIMENTAL DENSITY OF (U, Gd)O2 AND (U, Er)O2
        Atomic Energy, Vol. 113, No. 6, April, 2013 (Russian Original Vol. 113, No. 6, December, 2012), UDC 621.039.542.342:544.8

        Args:
        
        gd_wt_fraction (_type_): float
            Gadolinia oxide weight fraction. Must be a floating number between 0 and 1.

        """
        results = 10.96 - 0.031*gd_wt_fraction*100
        return results
        
    def uo2_er_density_calculator(self, er_wt_fraction):
        """
        This function will evaluate the density of (U-Er)O_2 mixture and return its value in g/cm3.
        The function uses a simplified model reported on the paper [1].
        
        [1] A. V. Fedotov, E. N. Mikheev, A. V. Lysikov, and V. V. Novikov, THEORETICAL AND EXPERIMENTAL DENSITY OF (U, Gd)O2 AND (U, Er)O2
        Atomic Energy, Vol. 113, No. 6, April, 2013 (Russian Original Vol. 113, No. 6, December, 2012), UDC 621.039.542.342:544.8

        Args:
        
        er_wt_fraction (_type_): float
            Erbia oxide weight fraction. Must be a floating number between 0 and 1.

        """
        results = 10.9616 - 0.0175*er_wt_fraction*100
        return results
        
    
        

class Fuel_Cycle_Costs:
    """
    This class performs fuel cycle cost evaluation.
    """
    def __init__(self):
        wip = 'WIP'