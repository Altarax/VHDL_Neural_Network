--! Import libraries
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

--! Import custom library
library work;
use work.types.all;

--! \brief Simple Neuron
--! \details
entity neuron is
	generic (
        TOTAL : integer := 2
	);
    port (
        clk     : in std_logic;
        reset   : in std_logic;

        inputs  : in int_array(0 to TOTAL);
        weigths : in int_array(0 to TOTAL);
        biais   : in integer;

        output  : out integer
    );
end entity neuron;

architecture rtl of neuron is

    signal sum              : integer := 0;
    signal sigmoid_output   : integer := 0;

begin

    sum_gen: process(clk, reset)
    begin
        if reset = '1' then
            sum <= 0;
        elsif rising_edge(clk) then
            for i in 0 to TOTAL loop
                sum <= sum + inputs(i)*weigths(i);
            end loop;
            sum <= sum + biais;
        end if;
    end process;

--    act_function: entity work.sigmoid 
--    port map (clock   => clk,
--              address => sum,
--              q       => sigmoid_output);
    
end architecture rtl;