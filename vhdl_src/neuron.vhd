--! Import libraries
library IEEE;
use IEEE.std_logic_1164.all;
use IEEE.numeric_std.all;

--! \brief Simple Neuron
--! \details
entity neuron is
	generic (
		INPUTS      : natural   := 2;
		WEIGHTS     : integer   := INPUTS+1
	);
    port (
        clk 	: in std_logic;
		reset	: in std_logic;
		start	: in std_logic;

		input	: in sfixed_bus_array(NEURON_INPUTS-1 downto 0);
		weights	: in sfixed_bus_array(NEURON_INPUTS-1 downto 0);

		output	: out sfixed(15 downto -frac);
		done	: out std_logic
    );
end entity neuron;

architecture rtl of neuron is
    
    signal pipeline_stage_s	: integer range 0 to 3                  := 0;
    signal mult_result_s 	: sfixed_bus_array(INPUTS-1 downto 0)   := (others => (others => 0));
    signal add_result_s 	: sfixed(15 downto -frac)               := (others => '0');
    
    signal act_func_input_s : sfixed(15 downto -frac)               := (others => '0');
    signal output_s         : sfixed(15 downto -frac)               := (others => '0');
    signal done_s           : std_logic                             := '0';

begin

    calculation:process(clk)
    begin
        if rising_edge(clk) then
            if reset = '1' then
                done_s <= '0';
                pipeline_stage_s <= -1;
                mult_result_s <= (others => (others => 0));
                add_result_s <= 0;
            else
				if start = '1' then
					pipeline_stage_s <= 0;
				else
					pipeline_stage_s <= pipeline_stage_s;
				end if;

                case pipeline_stage_s is

                    when 0 =>
                        -- Multiplication stage
						done_s <= '0';
                        for i in 0 to INPUTS-1 loop
                            mult_result_s(i) <= to_integer(unsigned(input(i))) * to_integer(unsigned(weights(i)));
                        end loop;
                        pipeline_stage_s <= 1;
                        
                    when 1 =>
                        -- Addition stage
                        for i in 0 to WEIGHTS-1 loop
                            if i = 0 then
                                add_result_s <= mult_result_s(i);
							else
								add_result_s <= add_result_s + mult_result_s(i);
							end if;
                        end loop;
                        pipeline_stage_s <= 2;
                        
                    when 2 =>
                        -- Act function
						pipeline_stage_s <= 3;

                    when 3 =>
                        -- Ack act function
                        pipeline_stage_s <= 0;
                        done_s <= '1';
						
                end case;
            end if;
        end if;

    end process;

    --! Activation function input generation
    act_func_input_s <= add_result_s when pipeline_stage_s = 2 else 0;

    --! Activation function instantiation
	act_func_inst: entity work.act_func()
    port map(
        clk    => clk,
        input  => act_func_input_s,
        output => output_s
    );

    --! Other connections
    output <= output_s;
    done <= done_s;
    
end architecture rtl;