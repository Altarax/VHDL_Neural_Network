--! Use simulation librairies
Library IEEE;
use IEEE.std_logic_TEXTIO.all;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

--! Import custom library
library work;
use work.types.all;

--! \brief Entity of neuron_tb
entity neuron_tb is
    --  Port ( );
end neuron_tb;

architecture simulation of neuron_tb is

    constant TOTAL : integer := 3;
    
    signal clk     : std_logic := '0';
    signal reset   : std_logic := '0';
    signal inputs  : int_array(0 to TOTAL);
    signal weigths : int_array(0 to TOTAL);
    signal biais   : integer := 0;
    signal output  : integer := 0;

begin
    
    UUT: entity work.neuron
        generic map (
            TOTAL => TOTAL
        )
        port map (
            clk => clk,
            reset => reset,
            inputs => inputs,
            weigths => weigths,
            biais => biais,
            output => output
        );

    reset_gen: process
        begin
            reset <= '0'; wait for 50ms;
        end process;

    clk_gen: process
        begin
            clk <= '1'; wait for 50ns;
            clk <= '0'; wait for 50ns;
        end process;

end simulation;