--! Import libraries
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;
use IEEE.fixed_float_types.all;
use IEEE.fixed_pkg.all; 

--! Type library
library work;
use work.types.all;

--! \brief Activation function
--! \details
entity act_funct is
    generic (
        NEURON_INPUTS : natural := 2
    );
    port (
        clk 	: in std_logic;
		reset	: in std_logic;
		
        input	: in sfixed_bus_array(NEURON_INPUTS-1 downto 0);
        output	: out sfixed(15 downto -frac)
    );
end entity neuron;

architecture rtl of act_func is

    signal result   : sfixed(NEURON_INPUTS-1 downto -frac);
    signal output_s : 
    
begin
    
    -- Return activation >= 0.5
    calculate_activation:process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then

            else
            end if;
        end if;
    end process;
    
end architecture rtl;